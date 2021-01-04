# TODO:
# fix best_average_reward and save_model etc.
# make loop to load saved model and test
# shuffle data during training, test on different data

import torch
from torch.optim import Adam
from typing import Tuple, Dict, List, Iterator
import numpy as np
import logging
from stable_baselines3 import PPO
import random

from seq2seq.helpers import log_parameters, pack_state_tensors, discount_cumsum
from seq2seq.rl_dataset import GroundedScanEnvironment
from seq2seq.rl_model import ModelRL, ActorCritic


logger = logging.getLogger("GroundedSCAN_learning")

use_cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PPOBuffer(object):
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, max_command_length: int, grid_size: int, grid_depth: int, action_dim: int, size: int,
                 gamma=0.99, gae_lambda=0.95):
        self.command_buffer = np.zeros([size, max_command_length], dtype=np.long)
        self.command_lengths_buffer = np.zeros([size], dtype=np.long)
        self.world_state_buffer = np.zeros([size, grid_size, grid_size, grid_depth], dtype=np.float32)
        self.action_buffer = np.zeros([size, action_dim], dtype=np.long)
        self.advantage_buffer = np.zeros(size, dtype=np.float32)
        self.reward_buffer = np.zeros(size, dtype=np.float32)
        self.return_buffer = np.zeros(size, dtype=np.float32)
        self.value_buffer = np.zeros(size, dtype=np.float32)
        self.logp_buffer = np.zeros(size, dtype=np.float32)
        self.gamma, self.gae_lambda = gamma, gae_lambda
        self.ptr, self.episode_start_idx, self.max_size = 0, 0, size

    def store(self, command: np.ndarray, command_lengths: np.int, world_state: np.ndarray, action: np.ndarray,
              reward: float, value: np.float32, logp: np.float32):
        """
        Append one time step of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.command_buffer[self.ptr] = command
        self.command_lengths_buffer[self.ptr] = command_lengths
        self.world_state_buffer[self.ptr] = world_state
        self.action_buffer[self.ptr] = action
        self.reward_buffer[self.ptr] = reward
        self.value_buffer[self.ptr] = value
        self.logp_buffer[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_value=0):
        """
        Call this at the end of an episode, or when an episode gets cut off
        by an epoch ending. Uses rewards and value estimates of current episode
        slice to compute advantage estimates with GAE-Lambda
        and returns (the targets for the value function) for each state.
        `last_value` should be 0 if the episode ended because the agent
        reached a terminal state (done), and otherwise V(s_T), the value
        function estimated for the last state.
        """

        episode_slice = slice(self.episode_start_idx, self.ptr)
        rewards = np.append(self.reward_buffer[episode_slice], last_value)
        values = np.append(self.value_buffer[episode_slice], last_value)

        # GAE-Lambda advantage calculation (per https://arxiv.org/pdf/1506.02438.pdf)
        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        self.advantage_buffer[episode_slice] = discount_cumsum(deltas, self.gamma * self.gae_lambda)

        # Returns is average of future discounted rewards
        self.return_buffer[episode_slice] = discount_cumsum(rewards, self.gamma)[:-1]

        self.episode_start_idx = self.ptr

    def get(self):
        """
        Returns all data currently in the buffer in tensors and resets the pointers.
        NB: note that advantages get normalized per batch and not here.
        """
        assert self.ptr == self.max_size, "Buffer has to be full before you can call .get."
        self.ptr, self.episode_start_idx = 0, 0
        data = dict(command=self.command_buffer,
                    command_length=self.command_lengths_buffer,
                    world_state=self.world_state_buffer,
                    action=self.action_buffer, returns=self.return_buffer,
                    advantage=self.advantage_buffer, logp=self.logp_buffer)
        types = dict(command=torch.long,
                     command_length=torch.long,
                     world_state=torch.float32,
                     action=torch.long,
                     returns=torch.float32,
                     advantage=torch.float32, logp=torch.float32)
        return {k: torch.as_tensor(v, dtype=types[k]) for k, v in data.items()}

    def get_batched_iterator(self, data: Dict[str, torch.tensor], batch_size: int) -> Iterator:
        """Randomly shuffle the data in the buffer and yield in batches."""
        random_permutation_data = torch.tensor(np.random.permutation(self.max_size))
        data_shuffled = {k: v.index_select(dim=0, index=random_permutation_data) for k, v in data.items()}
        (command, command_length, world_state, action, advantage,
         logp_old, ret) = data_shuffled["command"], data_shuffled["command_length"], data_shuffled["world_state"], \
                          data_shuffled["action"], data_shuffled["advantage"], data_shuffled["logp"], \
                          data_shuffled["returns"]
        for i in range(0, self.max_size, batch_size):
            yield dict(command=command[i:i + batch_size], command_length=command_length[i:i + batch_size],
                       world_state=world_state[i:i + batch_size], action=action[i:i + batch_size],
                       advantage=advantage[i:i + batch_size], logp=logp_old[i:i + batch_size],
                       returns=ret[i:i + batch_size])


class PPO(object):
    """..."""

    def __init__(self, actor_critic: ActorCritic, betas: Tuple[float, float],
                 gamma=0.99, pi_lr=3e-4, vf_coef=1., clip_ratio=0.2, target_kl=0.01, log_every=100):
        self.clip_ratio = clip_ratio
        self.vf_coef = vf_coef
        self.betas = betas
        self.gamma = gamma
        self.target_kl = target_kl
        self.log_every = log_every

        self.policy = actor_critic
        log_parameters(actor_critic)
        trainable_parameters = [parameter for parameter in actor_critic.parameters() if parameter.requires_grad]
        self.pi_vf_optimizer = Adam(trainable_parameters, lr=pi_lr, betas=betas)

    def compute_loss_pi_v(self, data: Dict[str, torch.tensor]) -> [torch.tensor, torch.tensor, dict]:
        """
        Calculate the PPO loss for the policy and the MSE for the value function.
        :param data: input batch of state, action, advantage, log_probs, and returns.

        :returns: policy loss, value loss, and mean entropy, clip fraction, and kl divergence.
        """
        (command, command_length, world_state, action, advantage,
         log_probs_old, returns) = data["command"], data["command_length"], data["world_state"], data["action"], \
                     data["advantage"], data["logp"], data["returns"]

        # Normalize the advantage.
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        # Forwards pass through the actor-critic network.
        pi, value, log_probs = self.policy.pi_and_v(command, command_length, world_state, action)

        # Calculate policy loss.
        policy_ratio = torch.exp(log_probs - log_probs_old)
        clipped_advantage = torch.clamp(policy_ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantage
        loss_pi = -(torch.min(policy_ratio * advantage, clipped_advantage)).mean()

        # Useful extra info: KL divergence between old and new policy, entropy of policy, and clipped fraction
        approx_kl = (log_probs_old - log_probs).mean().item()
        entropy = pi.entropy().mean().item()
        clipped = policy_ratio.gt(1 + self.clip_ratio) | policy_ratio.lt(1 - self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, entropy=entropy, clipfrac=clipfrac)

        # Calculate MSE of values.
        loss_v = ((value - returns) ** 2).mean()

        return loss_pi, loss_v, pi_info

    def update(self, buffer: PPOBuffer, train_pi_iters: int, training_batch_size: int) -> Tuple[float, float,
                                                                                                float, float]:
        data = buffer.get()

        with torch.no_grad():
            pi_l_old, v_l_old, pi_info_old = self.compute_loss_pi_v(data)
            pi_l_old, v_l_old = pi_l_old.item(), v_l_old.item()
        av_loss_pi = 0
        av_loss_v = 0
        num_updates = 0

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            approx_kl_divergences = []
            for batch in buffer.get_batched_iterator(data, training_batch_size):
                self.pi_vf_optimizer.zero_grad()
                loss_pi, loss_v, pi_info = self.compute_loss_pi_v(batch)
                kl = pi_info["kl"]
                approx_kl_divergences.append(kl)
                av_loss_pi += loss_pi.item()
                av_loss_v += loss_v.item()
                num_updates += 1
                loss = loss_pi + self.vf_coef * loss_v
                loss.backward()
                self.pi_vf_optimizer.step()
            if np.mean(np.array(approx_kl_divergences)) > 1.5 * self.target_kl:
                logger.info("Early stopping  due to reaching max kl at step %d of %d." %
                            (i, train_pi_iters))
                break
        return pi_l_old, v_l_old, av_loss_pi / num_updates, av_loss_v / num_updates


def evaluate(initial_observations: List[Tuple[dict, np.ndarray, np.ndarray]], policy: ActorCritic,
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
            input_command_t, command_lengths, world_state_t = pack_state_tensors((input_command, world_state))
            action, attention_weights = policy.act(input_command_t, command_lengths, world_state_t)
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
              pi_lr: float, adam_beta_1: float, training_batch_size: int,
              adam_beta_2: float, hidden_size: int, embedding_dimension: int, encoder_hidden_size: int,
              num_encoder_layers: int, encoder_bidirectional: bool, num_decoder_layers: int, cnn_dropout_p: float,
              decoder_dropout_p: float, train_pi_iters: int, cnn_kernel_size: int, cnn_hidden_num_channels: int,
              encoder_dropout_p: float, auxiliary_task: bool, conditional_attention: bool, attention_type: str,
              evaluate_every_epoch: int, total_timesteps: int,
              train_v_iters: int, ppo_log_every: int, visualize_trajectory_every: int, output_directory: str,
              gamma=0.99, lam=0.97, target_kl=0.01, clip_ratio=0.2, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
              seed=1234, **kwargs):
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
    state = training_env.reset()

    buffer = PPOBuffer(training_env.max_input_length, training_env.grid_dimensions[0],
                       training_env.grid_dimensions[-1], action_dim=1, size=timesteps_per_epoch,
                       gamma=gamma, gae_lambda=lam)

    # Initialize PPO algorithm
    ppo = PPO(actor_critic, betas, gamma, pi_lr, vf_coef, clip_ratio, target_kl, ppo_log_every)

    for i_epoch in range(num_epochs):
        for time_step in range(timesteps_per_epoch):
            # Unpack the state into tensors
            input_command_t, input_command_lengths, world_state_t = pack_state_tensors(state)
            action, value, log_probs, attn = ppo.policy.step(input_command_t, input_command_lengths, world_state_t)
            next_state, reward, done, _ = training_env.step(action.item() + 3)
            episode_reward += reward
            episode_length += 1

            # Saving reward and is_terminal:
            buffer.store(np.array(state[0]), len(state[0]), np.array(state[1]), action, reward, value,
                         log_probs[action.item()])

            state = next_state

            # Update if its time
            timeout = episode_length == max_episode_length
            terminal = done or timeout
            epoch_ended = time_step == timesteps_per_epoch - 1

            if terminal or epoch_ended:
                # If trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    _, value, _, _ = ppo.policy.step(
                        input_command_t, input_command_lengths, world_state_t)
                else:
                    value = torch.tensor(0)
                buffer.finish_path(value.item())
                if terminal:
                    average_episode_length += episode_length
                average_episode_reward += episode_reward
                num_episodes_terminated += 1
                if done:
                    num_episodes_done += 1
                state = training_env.reset()
                episode_reward, episode_length = 0, 0

        # Perform PPO update.
        pi_l_old, v_l_old, av_loss_pi, av_loss_v = ppo.update(buffer, train_pi_iters, training_batch_size)
        average_pi_loss += av_loss_pi
        average_v_loss += av_loss_v
        num_updates += 1
        logger.info("Epoch %d. Read dataset %d times. Percentage progress current epoch: %3.2f" % (
        i_epoch, training_env.read_full_training_set, training_env.percentage_progress
        ))
        logger.info("Epoch %d. Num. ep. done/terminated: %d/%d, Av. ep. length: %d, av. ep. reward: %5.2f" % (
        i_epoch, num_episodes_done, num_episodes_terminated,
        average_episode_length / max(num_episodes_terminated, 1),
        average_episode_reward / max(num_episodes_terminated, 1)))
        logger.info("Epoch %d. Loss pi: %5.4f, Average loss pi: %5.4f." % (i_epoch, av_loss_pi,
        average_pi_loss / num_updates))
        logger.info("Epoch %d. Loss v: %5.4f, Average loss v: %5.4f." % (i_epoch, av_loss_v,
                                                                         average_v_loss / num_updates))
        if (i_epoch + 1) % evaluate_every_epoch == 0:
            evaluation_observations = training_env.get_evaluation_observations()
            rewards_per_episode, episode_lengths, num_done = evaluate(evaluation_observations,
                                                                      ppo.policy, training_env, i_epoch,
                                                                      max_episode_length)
            logger.info("Evaluation Epoch %d. Num. ep. terminated: %d/%d, Av. ep. length: %d, av. ep. reward: %5.2f" % (
                i_epoch, num_done, len(evaluation_observations), np.mean(np.array(episode_lengths)),
                np.mean(np.array(rewards_per_episode))))
        # if np.mean(np.array(rewards_per_episode)) >= 1:
        #     stop_training = True
        average_episode_length = 0
        average_episode_reward = 0
        num_episodes_terminated = 0
        num_episodes_done = 0

