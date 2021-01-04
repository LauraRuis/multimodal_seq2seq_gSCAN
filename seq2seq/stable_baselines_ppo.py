# TODO:
# fix best_average_reward and save_model etc.
# make loop to load saved model and test
# shuffle data during training, test on different data

import torch
from torch.optim import Adam
from torch.nn import functional as F
import scipy.signal
from typing import Tuple, Dict, List, Iterator, Any, Optional
import numpy as np
import time
from collections import deque
import logging
from mpi4py import MPI
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
import random

from seq2seq.helpers import log_parameters, pack_state_tensors, explained_variance, safe_mean
from seq2seq.rl_dataset import GroundedScanEnvironment
from seq2seq.rl_model import ModelRL, ActorCritic


class MyFilter(logging.Filter):
    def filter(self, record):
        print(record.msg)
        if "STREAM" in record.msg:
            return False


myfilter = MyFilter()
logger = logging.getLogger()
logger.addFilter(myfilter)
logger = logging.getLogger("GroundedSCAN_learning")

use_cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def setup_pytorch_for_mpi():
    """
    Avoid slowdowns caused by each separate process's PyTorch using
    more than its fair share of CPU resources.
    """
    #print('Proc %d: Reporting original number of Torch threads as %d.'%(proc_id(), torch.get_num_threads()), flush=True)
    if torch.get_num_threads()==1:
        return
    fair_num_threads = max(int(torch.get_num_threads() / num_procs()), 1)
    torch.set_num_threads(fair_num_threads)
    #print('Proc %d: Reporting new number of Torch threads as %d.'%(proc_id(), torch.get_num_threads()), flush=True)


def broadcast(x, root=0):
    MPI.COMM_WORLD.Bcast(x, root=root)


def sync_params(module):
    """ Sync all parameters of module across all MPI processes. """
    if num_procs()==1:  # TODO: check
        return
    for p in module.parameters():
        p_numpy = p.data.numpy()
        broadcast(p_numpy)


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0,
         x1,
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def all_reduce(*args, **kwargs):
    return MPI.COMM_WORLD.Allreduce(*args, **kwargs)


def num_procs():
    """Count active MPI processes."""
    return MPI.COMM_WORLD.Get_size()


def mpi_avg(x):
    """Average a scalar or vector over MPI processes."""
    return mpi_sum(x) / num_procs()


def mpi_op(x, op):
    x, scalar = ([x], True) if np.isscalar(x) else (x, False)
    x = np.asarray(x, dtype=np.float32)
    buff = np.zeros_like(x, dtype=np.float32)
    all_reduce(x, buff, op=op)
    return buff[0] if scalar else buff


def mpi_sum(x):
    return mpi_op(x, MPI.SUM)


def mpi_statistics_scalar(x, with_min_and_max=False):
    """
    Get mean/std and optional min/max of scalar x across MPI processes.
    Args:
        x: An array containing samples of the scalar to produce statistics
            for.
        with_min_and_max (bool): If true, return min and max of x in
            addition to mean and std.
    """
    x = np.array(x, dtype=np.float32)
    global_sum, global_n = mpi_sum([np.sum(x), len(x)])
    mean = global_sum / global_n

    global_sum_sq = mpi_sum(np.sum((x - mean)**2))
    std = np.sqrt(global_sum_sq / global_n)  # compute global std

    if with_min_and_max:
        global_min = mpi_op(np.min(x) if len(x) > 0 else np.inf, op=MPI.MIN)
        global_max = mpi_op(np.max(x) if len(x) > 0 else -np.inf, op=MPI.MAX)
        return mean, std, global_min, global_max
    return mean, std


def mpi_avg_grads(module):
    """ Average contents of gradient buffers across MPI processes. """
    if num_procs() == 1:
        return
    for p in module.parameters():  # TODO: check whether to maybe add multiple modules
        p_grad_numpy = p.grad.numpy()   # numpy view of tensor data
        avg_p_grad = mpi_avg(p.grad)
        p_grad_numpy[:] = avg_p_grad[:]


class RolloutBuffer(object):
    """
    ...
    """
    def __init__(self, max_command_length: int, grid_size: int, grid_depth: int, action_dim: int, size: int,
                 gamma=0.99, lam=0.95):
        self.command_observation_buffer = None
        self.command_lengths_buffer = None
        self.world_state_observation_buffer = None
        self.action_buffer = None
        self.advantage_buffer = None
        self.reward_buffer = None
        self.return_buffer = None
        self.value_buffer = None
        self.logp_buffer = None
        self.done_buffer = None
        self.gamma, self.gae_lambda = gamma, lam

        self.max_command_length = max_command_length
        self.grid_size = grid_size
        self.grid_depth = grid_depth
        self.action_dim = action_dim
        self.buffer_size = size
        self.pos = 0
        self.path_start_idx = 0
        self.full = False
        self.generator_ready = False

    def reset(self):
        self.full = False
        self.generator_ready = False
        self.pos = 0
        self.path_start_idx = 0
        self.command_observation_buffer = np.zeros([self.buffer_size, 1, self.max_command_length], dtype=np.long)
        self.command_lengths_buffer = np.zeros([self.buffer_size, 1], dtype=np.long)
        self.world_state_observation_buffer = np.zeros([self.buffer_size, 1, self.grid_size, self.grid_size,
                                                        self.grid_depth], dtype=np.float32)
        self.action_buffer = np.zeros([self.buffer_size, 1, self.action_dim], dtype=np.long)
        self.advantage_buffer = np.zeros([self.buffer_size, 1], dtype=np.float32)
        self.reward_buffer = np.zeros([self.buffer_size, 1], dtype=np.float32)
        self.return_buffer = np.zeros([self.buffer_size, 1], dtype=np.float32)
        self.value_buffer = np.zeros([self.buffer_size, 1], dtype=np.float32)
        self.logp_buffer = np.zeros([self.buffer_size, 1], dtype=np.float32)
        self.done_buffer = np.zeros([self.buffer_size, 1], dtype=np.float32)

    def compute_returns_and_advantage(self, last_values: torch.Tensor) -> None:
        """
        Post-processing step: compute the returns (sum of discounted rewards)
        and GAE advantage.
        Adapted from Stable-Baselines PPO2.
        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain vanilla advantage (A(s) = R - V(S))
        where R is the discounted reward with value bootstrap,
        set ``gae_lambda=1.0`` during initialization.
        :param last_values:
        :param dones:
        """
        last_values = last_values.clone().cpu().numpy().flatten()
        last_gae_lam = 0
        episode_slice = slice(self.path_start_idx, self.pos)
        for step in range(self.pos - 1, self.path_start_idx - 1, -1):
            if step == self.pos - 1:
                next_values = last_values
            else:
                next_values = self.value_buffer[step + 1]
            delta = self.reward_buffer[step] + self.gamma * next_values - self.value_buffer[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * last_gae_lam
            self.advantage_buffer[step] = last_gae_lam
        self.return_buffer[episode_slice] = self.advantage_buffer[episode_slice] + self.value_buffer[episode_slice]
        self.path_start_idx = self.pos

    def add(self, command_obs: np.ndarray, command_length_obs: np.ndarray, world_state_obs: np.ndarray,
            action: np.ndarray, reward: np.ndarray, done: np.ndarray, value: torch.Tensor,
            log_prob: torch.Tensor) -> None:
        """
        :param command_obs: Observation
        :param command_length_obs: Observation
        :param world_state_obs: Observation
        :param action: Action
        :param reward:
        :param done: End of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        self.command_observation_buffer[self.pos] = np.array(command_obs).copy()
        self.command_lengths_buffer[self.pos] = np.array(command_length_obs).copy()
        self.world_state_observation_buffer[self.pos] = np.array(world_state_obs).copy()
        self.action_buffer[self.pos] = np.array(action).copy()
        self.reward_buffer[self.pos] = np.array(reward).copy()
        self.done_buffer[self.pos] = np.array(done).copy()
        self.value_buffer[self.pos] = value.clone().cpu().numpy().flatten()
        self.logp_buffer[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)
        :param arr:
        :return:
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = shape + (1,)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def get(self, batch_size: Optional[int] = None) -> Iterator:
        assert self.full, "Can only get buffer if full."
        indices = np.random.permutation(self.buffer_size * 1)
        # Prepare the data
        if not self.generator_ready:
            for tensor in ["command_observation_buffer", "command_lengths_buffer", "world_state_observation_buffer",
                           "action_buffer", "value_buffer", "logp_buffer", "advantage_buffer", "return_buffer"]:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size

        self.advantage_buffer = (self.advantage_buffer - self.advantage_buffer.mean()) / (
                self.advantage_buffer.std() + 1e-8)
        start_idx = 0
        while start_idx < self.buffer_size:
            yield self._get_samples(indices[start_idx: start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray) -> dict:
        data = dict(command=self.command_observation_buffer[batch_inds],
                    command_length=self.command_lengths_buffer[batch_inds],
                    world_state=self.world_state_observation_buffer[batch_inds],
                    actions=self.action_buffer[batch_inds], returns=self.return_buffer[batch_inds],
                    advantages=self.advantage_buffer[batch_inds], log_probs=self.logp_buffer[batch_inds])
        types = dict(command=torch.long,
                     command_length=torch.long,
                     world_state=torch.float32,
                     actions=torch.long,
                     returns=torch.float32,
                     advantages=torch.float32, log_probs=torch.float32)
        return {k: torch.as_tensor(v, dtype=types[k]) for k, v in data.items()}


class PPOBuffer(object):
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.

    Copied from SpinningUp Library by OpenAI.
    """

    def __init__(self, max_command_length: int, grid_size: int, grid_depth: int, action_dim: int, size: int,
                 gamma=0.99, lam=0.95):
        self.command_observation_buffer = np.zeros([size, max_command_length], dtype=np.long)
        self.command_lengths_buffer = np.zeros([size], dtype=np.long)
        self.world_state_observation_buffer = np.zeros([size, grid_size, grid_size, grid_depth], dtype=np.float32)
        self.action_buffer = np.zeros([size, action_dim], dtype=np.long)
        self.advantage_buffer = np.zeros(size, dtype=np.float32)
        self.reward_buffer = np.zeros(size, dtype=np.float32)
        self.rewards_togo_buffer = np.zeros(size, dtype=np.float32)
        self.value_buffer = np.zeros(size, dtype=np.float32)
        self.logp_buffer = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, command: np.ndarray, command_lengths: np.int,
              world_state: np.ndarray, action: np.ndarray,
              reward: float, value: np.float32, logp: np.float32):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.command_observation_buffer[self.ptr] = command
        self.command_lengths_buffer[self.ptr] = command_lengths
        self.world_state_observation_buffer[self.ptr] = world_state
        self.action_buffer[self.ptr] = action
        self.reward_buffer[self.ptr] = reward
        self.value_buffer[self.ptr] = value
        self.logp_buffer[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_value=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.
        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rewards = np.append(self.reward_buffer[path_slice], last_value)
        values = np.append(self.value_buffer[path_slice], last_value)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        self.advantage_buffer[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.rewards_togo_buffer[path_slice] = discount_cumsum(rewards, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        advantage_mean, advantage_std = mpi_statistics_scalar(self.advantage_buffer)
        if advantage_std > 0:  # TODO: fix better
            self.advantage_buffer = (self.advantage_buffer - advantage_mean) / advantage_std
        data = dict(command=self.command_observation_buffer,
                    command_length=self.command_lengths_buffer,
                    world_state=self.world_state_observation_buffer,
                    action=self.action_buffer, rewards_togo=self.rewards_togo_buffer,
                    advantage=self.advantage_buffer, logp=self.logp_buffer)
        types = dict(command=torch.long,
                     command_length=torch.long,
                     world_state=torch.float32,
                     action=torch.long,
                     rewards_togo=torch.float32,
                     advantage=torch.float32, logp=torch.float32)
        return {k: torch.as_tensor(v, dtype=types[k]) for k, v in data.items()}

    def get_batched_iterator(self, data: Dict[str, torch.tensor], batch_size: int) -> Iterator:
        random_permutation_data = torch.tensor(np.random.permutation(self.max_size))
        data_shuffled = {k: v.index_select(dim=0, index=random_permutation_data) for k, v in data.items()}
        (command, command_length, world_state, action, advantage,
         logp_old, ret) = data_shuffled["command"], data_shuffled["command_length"], data_shuffled["world_state"], \
                          data_shuffled["action"], data_shuffled["advantage"], data_shuffled["logp"], \
                          data_shuffled["rewards_togo"]
        for i in range(0, self.max_size, batch_size):
            yield dict(command=command[i:i + batch_size], command_length=command_length[i:i + batch_size],
                       world_state=world_state[i:i + batch_size], action=action[i:i + batch_size],
                       advantage=advantage[i:i + batch_size], logp=logp_old[i:i + batch_size],
                       rewards_togo=ret[i:i + batch_size])


class PPO(object):
    """..."""

    def __init__(self, actor_critic, betas, gamma=0.99, pi_lr=3e-4, vf_lr=1e-3, clip_ratio=0.2,
                 target_kl=0.01, log_every=100):
        self.pi_lr = pi_lr
        self.vf_lr = vf_lr
        self.clip_ratio = clip_ratio
        self.betas = betas
        self.gamma = gamma
        self.target_kl = target_kl
        self.log_every = log_every

        self.policy = actor_critic

        log_parameters(actor_critic)
        trainable_parameters = [parameter for parameter in actor_critic.parameters() if parameter.requires_grad]

        self.pi_vf_optimizer = Adam(trainable_parameters, lr=pi_lr, betas=betas)

    def compute_loss_pi(self, data: Dict[str, torch.tensor]) -> Tuple[torch.tensor, dict]:
        """

        :param data: ..
        """
        (command, command_length, world_state, action, advantage,
         logp_old) = data["command"], data["command_length"], data["world_state"], data["action"], \
                     data["advantage"], data["logp"]

        # Policy loss
        pi, logp = self.policy.pi(command, command_length, world_state, action)  # batch
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantage
        loss_pi = -(torch.min(ratio * advantage, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1 + self.clip_ratio) | ratio.lt(1 - self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    def compute_loss_v(self, data: Dict[str, torch.tensor]) -> torch.tensor:
        """

        :param data:
        """
        (command, command_length, world_state, ret) = data["command"], data["command_length"], \
                                                      data["world_state"], data["rewards_togo"]
        return ((self.policy.v(command, command_length, world_state) - ret) ** 2).mean()

    def compute_loss_pi_v(self, data: Dict[str, torch.tensor]) -> [torch.tensor, torch.tensor, dict]:
        """

        :param data: ..
        """
        (command, command_length, world_state, action, advantage,
         logp_old, ret) = data["command"], data["command_length"], data["world_state"], data["action"], \
                     data["advantage"], data["logp"], data["rewards_togo"]

        # Policy loss
        pi, v, logp = self.policy.pi_and_v(command, command_length, world_state, action)  # batch
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantage
        loss_pi = -(torch.min(ratio * advantage, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1 + self.clip_ratio) | ratio.lt(1 - self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        loss_v = ((v - ret) ** 2).mean()

        return loss_pi, loss_v, pi_info

    def update(self, buffer: PPOBuffer, train_pi_iters: int, training_batch_size: int) -> Tuple[float, float,
                                                                                                float, float]:
        # logger.info("Updating.")
        data = buffer.get()

        with torch.no_grad():
            pi_l_old, v_l_old, pi_info_old = self.compute_loss_pi_v(data)
            pi_l_old, v_l_old = pi_l_old.item(), v_l_old.item()
        av_loss_pi = 0
        av_loss_v = 0
        num_updates = 0
        num_early_stopping = 0
        num_not_early_stopping = 0

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            for batch in buffer.get_batched_iterator(data, training_batch_size):
                self.pi_vf_optimizer.zero_grad()
                loss_pi, loss_v, pi_info = self.compute_loss_pi_v(batch)
                kl = mpi_avg(pi_info["kl"])
                if kl > 1.5 * self.target_kl:
                    num_early_stopping += 1
                    break
                else:
                    num_not_early_stopping += 1
                av_loss_pi += loss_pi.item()
                av_loss_v += loss_v.item()
                num_updates += 1
                loss = loss_pi + loss_v
                loss.backward()
                mpi_avg_grads(self.policy.pi)    # average grads across MPI processes
                mpi_avg_grads(self.policy.v)
                self.pi_vf_optimizer.step()
        if num_early_stopping > 0:
            logger.info("Early stopping  due to reaching max kl %d/%d times." %
                        (num_early_stopping, num_early_stopping + num_not_early_stopping))
        return pi_l_old, v_l_old, av_loss_pi / num_updates, av_loss_v / num_updates


class PPONew(object):
    """..."""

    def __init__(self, policy: ActorCritic, env: GroundedScanEnvironment, betas: Tuple[float, float], batch_size: int,
                 rollout_buffer: RolloutBuffer, num_steps: int, ent_coef: float, vf_coef: float, max_grad_norm: float,
                 evaluate_every_epoch: int, num_epochs: int, max_episode_length: int, gamma=0.99, pi_lr=3e-4,
                 vf_lr=1e-3, clip_ratio=0.2, target_kl=0.01, log_every=100):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_steps = num_steps
        self.pi_lr = pi_lr
        self.vf_lr = vf_lr
        self.clip_ratio = clip_ratio
        self.betas = betas
        self.gamma = gamma
        self.target_kl = target_kl
        self.log_every = log_every
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        self.rollout_buffer = rollout_buffer
        self.policy = policy
        self.env = env

        # Initialize internal states.
        self.num_timesteps = 0
        self.max_episode_length = max_episode_length
        self._n_updates = 0
        self._last_state = env.reset()
        self._last_done = np.zeros((1,), dtype=np.bool)
        self.start_time = time.time()
        self.average_episode_reward = 0
        self.average_episode_length = 0
        self.num_updated = 0
        self.num_done = 0
        self.evaluate_every_epoch = evaluate_every_epoch

        # Track the training progress remaining (from 1 to 0)
        # this is used to update the learning rate
        self._current_progress_remaining = 1

    def _update_info_buffer(self, reward: float, ep_length: int, done: bool) -> None:
        """
        Retrieve reward and episode length and update the buffer
        if using Monitor wrapper.
        :param infos:
        """
        self.average_episode_reward += reward
        self.average_episode_length += ep_length
        self.num_done += 1 if done else 0
        self.num_updated += 1

    def collect_rollouts(self, env: GroundedScanEnvironment, rollout_buffer: RolloutBuffer,
                         num_rollout_steps: int) -> bool:
        assert self._last_state is not None, "No previous observation was provided"
        num_steps, next_state, done = 0, None, False
        self.num_done = 0
        self.average_episode_reward = 0
        self.average_episode_length = 0
        self.num_updated = 0
        env.set_progress_reward()
        rollout_buffer.reset()
        episode_length = 0
        episode_reward = 0
        episode_attn = []
        while num_steps < num_rollout_steps:
            input_command_t, input_command_lengths, world_state_t = pack_state_tensors(self._last_state)
            action, value, log_probs, attn = self.policy.step(input_command_t, input_command_lengths, world_state_t)
            episode_attn.append(attn)
            next_state, reward, done, _ = env.step(action.item() + 3)

            self.num_timesteps += 1
            episode_length += 1
            episode_reward += reward
            num_steps += 1

            rollout_buffer.add(self._last_state[0], np.array([len(self._last_state[0])]), self._last_state[1], action,
                               np.array(reward), np.array(self._last_done), torch.tensor(value),
                               log_probs[action.item()])
            if self._last_done or episode_length == self.max_episode_length or num_steps == num_rollout_steps:
                self._update_info_buffer(reward=episode_reward, ep_length=episode_length, done=self._last_done)
                if self._last_done:
                    rollout_buffer.compute_returns_and_advantage(last_values=torch.tensor(0))
                else:
                    rollout_buffer.compute_returns_and_advantage(last_values=torch.tensor(value))
                # self.env.dataset.visualize_current_sequence(
                #     parent_save_dir="training_episodes",
                #     attention_weights=episode_attn)
                self._last_state = env.reset()
                self._last_done = False
                episode_length, episode_reward = 0, 0
                episode_attn.clear()
            else:
                self._last_state = next_state
                self._last_done = done
        assert next_state is not None, "Next state did not get filled."
        assert done is not None, "Done did not get filled."
        return True

    def _update_current_progress_remaining(self, num_timesteps: int, total_timesteps: int) -> None:
        """
        Compute current progress remaining (starts from 1 and ends to 0)
        :param num_timesteps: current number of timesteps
        :param total_timesteps:
        """
        self._current_progress_remaining = 1.0 - float(num_timesteps) / float(total_timesteps)

    def learn(self, total_timesteps: int, log_interval=1):
        iteration = 0
        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(self.env, self.rollout_buffer, num_rollout_steps=self.num_steps)
            if not continue_training:
                break
            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                fps = int(self.num_timesteps / (time.time() - self.start_time))
                logger.info("time/iterations %d" % iteration)
                logger.info("rollout/ep_rew_mean %5.2f" % (self.average_episode_reward / self.num_updated))
                logger.info("rollout/ep_len_mean %5.2f" % (self.average_episode_length / self.num_updated))
                logger.info("rollout/done_vs_terminated %d/%d" % (self.num_done, self.num_updated))
                logger.info("rollout/num_saw_full_training_set %d" % self.env.read_full_training_set)
                logger.info("time/fps %d" % fps)
                logger.info("time/time_elapsed %5.2f" % int(time.time() - self.start_time))
                logger.info("time/total_timesteps %d" % self.num_timesteps)
                # logger.dump(step=self.num_timesteps)
            # TODO: log progress
            self.train()

            if (iteration + 1) % self.evaluate_every_epoch == 0:
                evaluation_observations = self.env.get_evaluation_observations()
                rewards_per_episode, episode_lengths, num_done = evaluate(evaluation_observations,
                                                                          self, self.env, iteration,
                                                                          self.max_episode_length)
                logger.info("Evaluation Iteration %d. Num. ep. terminated: %d/%d" % (
                    iteration, num_done, len(evaluation_observations)))

    def compute_loss_pi_v(self, data: Dict[str, torch.tensor]) -> [torch.tensor, torch.tensor, dict]:
        """

        :param data: ..
        """
        (command, command_length, world_state, action, advantage,
         logp_old, ret) = data["command"], data["command_length"], data["world_state"], data["actions"], \
                     data["advantages"], data["log_probs"], data["returns"]

        # Policy loss
        pi, v, logp = self.policy.pi_and_v(command, command_length.squeeze(), world_state, action.squeeze())  # batch
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantage
        loss_pi = -(torch.min(ratio * advantage, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1 + self.clip_ratio) | ratio.lt(1 - self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        loss_v = ((v - ret) ** 2).mean()

        return loss_pi, loss_v, pi_info

    def train(self):

        entropy_losses, all_kl_divs = [], []
        pg_losses, value_losses = [], []
        clip_fractions = []

        for epoch in range(self.num_epochs):
            approx_kl_divergences = []

            for rollout_data in self.rollout_buffer.get(self.batch_size):
                # Optimization step
                # self.policy.optimizer.zero_grad()
                actions = rollout_data["actions"]
                pi, values, log_probs = self.policy.pi_and_v(rollout_data["command"],
                                                             rollout_data["command_length"].squeeze(),
                                                             rollout_data["world_state"], actions.squeeze())
                # loss_pi, loss_v, pi_info = self.compute_loss_pi_v(rollout_data)
                entropy = pi.entropy()
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data["advantages"]
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = torch.exp(log_probs - rollout_data["log_probs"])

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)  # TODO: check if this is the right clip quantity
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = torch.mean((torch.abs(ratio - 1) > self.clip_ratio).float()).item()
                clip_fractions.append(clip_fraction)

                values_pred = values
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data["returns"].squeeze(), values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -torch.mean(-log_probs)
                else:
                    entropy_loss = -torch.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                approx_kl_divergence = torch.mean(rollout_data["log_probs"] - log_probs).detach().cpu().numpy()
                approx_kl_divergences.append(approx_kl_divergence)
                # if self.target_kl is not None and pi_info["kl"] > 1.5 * self.target_kl:
                #     print(
                #         f"Early stopping at step {epoch} due to reaching max kl: {np.mean(approx_kl_divergences):.2f}")
                #     break

                loss.backward()
                # Clip grad norm
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            all_kl_divs.append(np.mean(approx_kl_divergences))
            if self.target_kl is not None and np.mean(approx_kl_divergences) > 1.5 * self.target_kl:
                print(f"Early stopping at step {epoch} due to reaching max kl: {np.mean(approx_kl_divergences):.2f}")
                break

        self._n_updates += self.num_epochs
        explained_var = explained_variance(self.rollout_buffer.value_buffer.flatten(),
                                           self.rollout_buffer.return_buffer.flatten())

        logger.info("train/entropy_loss %5.5f" % np.mean(entropy_losses))
        logger.info("train/policy_gradient_loss %5.5f" % np.mean(pg_losses))
        logger.info("train/value_loss %5.5f" % np.mean(value_losses))
        logger.info("train/approx_kl %5.5f" % np.mean(approx_kl_divergences))
        logger.info("train/clip_fraction %5.5f" % np.mean(clip_fractions))
        logger.info("train/loss %5.5f" % loss.item())
        logger.info("train/explained_variance %5.5f" % explained_var)
        logger.info("train/n_updates %d" % self._n_updates)
        logger.info("train/clip_range %5.5f" % self.clip_ratio)

    def predict(self, state: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        input_command_t, input_command_lengths, world_state_t = pack_state_tensors(state)
        action, attention_weights = self.policy.act(input_command_t, input_command_lengths, world_state_t)
        return action, attention_weights


def evaluate(initial_observations: List[Tuple[dict, np.ndarray, np.ndarray]], ppo: PPONew,
             env: GroundedScanEnvironment, epoch: int, max_steps_per_episode=50):
    env.unset_progress_reward()
    rewards_per_episode = []
    episode_lengths = []
    num_done = 0
    done = False
    for example, input_command, world_state in initial_observations:
        env.dataset.initialize_rl_example(example)
        episode_reward = 0
        episode_length = 0
        attention_weights_episode = []
        for timestep in range(max_steps_per_episode):
            # Get an action and value from the current policy
            action, attention_weights = ppo.predict((input_command, world_state))
            attention_weights_episode.append(attention_weights.tolist())
            next_state, reward, done, _ = env.step(action.item() + 3)
            episode_reward += reward
            episode_length += 1

            input_command, world_state = next_state

            if done:
                num_done += 1
                break
        if done:
            env.dataset.visualize_current_sequence(parent_save_dir="validation_episodes_done_epoch_{}".format(epoch),
                                                   attention_weights=attention_weights_episode)
        else:
            env.dataset.visualize_current_sequence(parent_save_dir="validation_episodes_failed_epoch_{}".format(epoch),
                                                   attention_weights=attention_weights_episode)
        rewards_per_episode.append(episode_reward)
        episode_lengths.append(episode_length)
    return rewards_per_episode, episode_lengths, num_done


def train_ppo(training_env: GroundedScanEnvironment, simple_situation_representation: bool, ppo_log_interval: int,
              max_episode_length: int, timesteps_per_epoch: int, num_epochs: int,
              pi_lr: float, vf_lr: float, adam_beta_1: float, training_batch_size: int,
              adam_beta_2: float, hidden_size: int, embedding_dimension: int, encoder_hidden_size: int,
              num_encoder_layers: int, encoder_bidirectional: bool, num_decoder_layers: int, cnn_dropout_p: float,
              decoder_dropout_p: float, train_pi_iters: int, cnn_kernel_size: int, cnn_hidden_num_channels: int,
              encoder_dropout_p: float, auxiliary_task: bool, conditional_attention: bool, attention_type: str,
              ppo_buffer_size: int, evaluate_every_epoch: int, total_timesteps: int,
              train_v_iters: int, ppo_log_every: int, visualize_trajectory_every: int, output_directory: str,
              gamma=0.99, lam=0.97, target_kl=0.01, clip_ratio=0.2, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
              seed=1234, **kwargs):

    setup_pytorch_for_mpi()

    # Random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    action_dim = training_env.action_space.n
    betas = (adam_beta_1, adam_beta_2)

    representation_model = ModelRL(input_vocabulary_size=training_env.input_vocabulary_size,
                                   embedding_dimension=embedding_dimension, encoder_hidden_size=encoder_hidden_size,
                                   num_encoder_layers=num_encoder_layers,
                                   target_vocabulary_size=training_env.target_vocabulary_size,
                                   encoder_dropout_p=encoder_dropout_p,
                                   encoder_bidirectional=encoder_bidirectional, num_decoder_layers=num_decoder_layers,
                                   decoder_dropout_p=decoder_dropout_p,
                                   hidden_size=hidden_size,
                                   num_cnn_channels=training_env.grid_dimensions[-1], cnn_kernel_size=cnn_kernel_size,
                                   cnn_dropout_p=cnn_dropout_p,
                                   cnn_hidden_num_channels=cnn_hidden_num_channels,
                                   input_padding_idx=training_env.input_vocabulary.pad_idx,
                                   target_pad_idx=training_env.target_vocabulary.pad_idx,
                                   target_eos_idx=training_env.target_vocabulary.eos_idx,
                                   output_directory=output_directory, conditional_attention=conditional_attention,
                                   auxiliary_task=auxiliary_task,
                                   simple_situation_representation=simple_situation_representation,
                                   attention_type=attention_type)
    actor_critic = ActorCritic(representation_model, action_dim, hidden_size, training_env.target_vocabulary.sos_idx,
                               pi_lr=pi_lr, betas=betas)

    # Initialize PPO algorithm
    # ppo = PPO(actor_critic, betas, gamma, pi_lr, vf_lr, clip_ratio, target_kl, ppo_log_every)

    # Initial observation
    # state = training_env.reset()

    # Logging variables
    episode_reward = 0
    episode_length = 0
    average_episode_reward = 0
    average_episode_length = 0
    num_episodes_terminated = 0
    num_episodes_done = 0
    best_average_reward = 0
    average_pi_loss = 0
    average_v_loss = 0
    num_updates = 0
    stop_training = False
    stop_visualizing = False
    done = False

    # buffer = PPOBuffer(training_env.max_input_length, training_env.grid_dimensions[0],
    #                    training_env.grid_dimensions[-1], action_dim=1, size=ppo_buffer_size,
    #                    gamma=gamma, lam=lam)

    rollout_buffer = RolloutBuffer(training_env.max_input_length, training_env.grid_dimensions[0],
                       training_env.grid_dimensions[-1], action_dim=1, size=timesteps_per_epoch,
                       gamma=gamma, lam=lam)

    ppo = PPONew(actor_critic, training_env, betas, rollout_buffer=rollout_buffer,
                 num_steps=timesteps_per_epoch, num_epochs=num_epochs, evaluate_every_epoch=evaluate_every_epoch,
                 ent_coef=ent_coef, batch_size=training_batch_size, max_episode_length=max_episode_length,
                 vf_coef=vf_coef, max_grad_norm=max_grad_norm, gamma=gamma, pi_lr=pi_lr, vf_lr=vf_lr,
                 clip_ratio=clip_ratio,
                 target_kl=target_kl, log_every=100)

    ppo.learn(total_timesteps=total_timesteps)
        #     if np.mean(np.array(rewards_per_episode)) >= 1:
        #         stop_training = True
        # for time_step in range(timesteps_per_epoch):
        #
        #     # Unpack the state into tensors
        #     input_command = state[0]
        #     world_state = state[1]
        #     input_command_t = torch.tensor(input_command, dtype=torch.long, device=device).unsqueeze(0)
        #     input_command_lengths = [len(input_command)]
        #     world_state_t = torch.tensor(world_state, dtype=torch.float32, device=device).unsqueeze(0)
        #
        #     # Get an action and value from the current policy
        #     action, value, log_probs = ppo.policy.step(input_command_t, input_command_lengths, world_state_t)
        #     next_state, reward, done, _ = training_env.step(action.item() + 3)
        #     episode_reward += reward
        #     episode_length += 1
        #
        #     # Saving reward and is_terminal:
        #     buffer.store(np.array(input_command), len(input_command), np.array(world_state), action, reward, value,
        #                  log_probs[action.item()])
        #
        #     state = next_state
        #
        #     # Update if its time
        #     timeout = episode_length == max_episode_length
        #     terminal = done or timeout
        #     epoch_ended = time_step == timesteps_per_epoch - 1
        #
        #     if terminal or epoch_ended:
        #         if epoch_ended and not terminal:
        #             logger.info("Warning: trajectory cut off by epoch at %d steps." % episode_length)
        #         # If trajectory didn't reach terminal state, bootstrap value target
        #         if timeout or epoch_ended:
        #             _, value, _ = ppo.policy.step(
        #                 input_command_t, input_command_lengths, world_state_t)
        #         else:
        #             value = torch.tensor(0)
        #         buffer.finish_path(value.item())
        #         if terminal:
        #             average_episode_length += episode_length
        #             average_episode_reward += episode_reward
        #             num_episodes_terminated += 1
        #             if done:
        #                 # logger.info("Episode ended. episode reward: %5.2f, episode length: %d" % (episode_reward,
        #                 #                                                                           episode_length))
        #                 # if not stop_visualizing:
        #                 #     training_env.dataset.visualize_current_sequence(parent_save_dir="ended_training_episodes")
        #                 num_episodes_done += 1
        #                 if average_episode_reward / max(num_episodes_terminated, 1) > 0.7:  # TODO: change
        #                     stop_visualizing = True
        #         state = training_env.reset()
        #         episode_reward = 0
        #         episode_length = 0
        #
        #     # if (time_step + 1) % ppo_log_interval == 0:
        #     #     logger.info("       Num. ep. done/terminated: %d/%d, Av. ep. length: %d, av. ep. reward: %5.2f" % (
        #     #         num_episodes_done, num_episodes_terminated,
        #     #         average_episode_length / max(num_episodes_terminated, 1),
        #     #         average_episode_reward / max(num_episodes_terminated, 1)))
        #
        # # Perform PPO update.
        # pi_l_old, v_l_old, av_loss_pi, av_loss_v = ppo.update(buffer, train_pi_iters, training_batch_size)
        # average_pi_loss += av_loss_pi
        # average_v_loss += av_loss_v
        # num_updates += 1
        # logger.info("Epoch %d. Read dataset %d times. Percentage progress current epoch: %3.2f" % (
        #     i_epoch, training_env.read_full_training_set, training_env.percentage_progress
        # ))
        # logger.info("Epoch %d. Num. ep. done/terminated: %d/%d, Av. ep. length: %d, av. ep. reward: %5.2f" % (
        #     i_epoch, num_episodes_done, num_episodes_terminated,
        #     average_episode_length / max(num_episodes_terminated, 1),
        #     average_episode_reward / max(num_episodes_terminated, 1)))
        # logger.info("Epoch %d. Loss pi: %5.4f, Average loss pi: %5.4f." % (i_epoch, av_loss_pi,
        #                                                                    average_pi_loss / num_updates))
        # logger.info("Epoch %d. Loss v: %5.4f, Average loss v: %5.4f." % (i_epoch, av_loss_v,
        #                                                                  average_v_loss / num_updates))
        # if (i_epoch + 1) % evaluate_every_epoch == 0:
        #     evaluation_observations = training_env.get_evaluation_observations()
        #     rewards_per_episode, episode_lengths, num_done = evaluate(evaluation_observations,
        #                                                               ppo.policy, training_env, i_epoch,
        #                                                               max_episode_length)
        #     logger.info("Evaluation Epoch %d. Num. ep. terminated: %d/%d, Av. ep. length: %d, av. ep. reward: %5.2f" % (
        #         i_epoch, num_done, len(evaluation_observations), np.mean(np.array(episode_lengths)),
        #         np.mean(np.array(rewards_per_episode))))
        #     # if np.mean(np.array(rewards_per_episode)) >= 1:
        #     #     stop_training = True
        # average_episode_length = 0
        # average_episode_reward = 0
        # num_episodes_terminated = 0
        # num_episodes_done = 0
