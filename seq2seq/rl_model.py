import torch
import logging
from typing import List, Dict, Tuple, Any
import numpy as np
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.distributions import Categorical
import os
import shutil

from seq2seq.cnn_model import ConvolutionalNet
from seq2seq.seq2seq_model import EncoderRNN
from seq2seq.seq2seq_model import Attention
from seq2seq.helpers import log_parameters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger("GroundedSCAN_learning")
use_cuda = True if torch.cuda.is_available() else False


class ModelRL(nn.Module):
    """The neural network to extract a multi-modal representation from a language command and world state."""

    def __init__(self, input_vocabulary_size: int, embedding_dimension: int, encoder_hidden_size: int,
                 num_encoder_layers: int, encoder_dropout_p: float,
                 encoder_bidirectional: bool,  hidden_size: int, num_cnn_channels: int, cnn_kernel_size: int,
                 cnn_dropout_p: float, cnn_hidden_num_channels: int, input_padding_idx: int, target_pad_idx: int,
                 target_eos_idx: int, output_directory: str, auxiliary_task: bool,
                 simple_situation_representation: bool, **kwargs):
        super(ModelRL, self).__init__()

        self.simple_situation_representation = simple_situation_representation
        cnn_input_channels = num_cnn_channels

        # Input: [batch_size, image_width, image_width, num_channels]
        # Output: [batch_size, image_width * image_width, num_conv_channels * 3]
        self.situation_encoder = ConvolutionalNet(num_channels=cnn_input_channels,
                                                  cnn_kernel_size=cnn_kernel_size,
                                                  num_conv_channels=cnn_hidden_num_channels,
                                                  dropout_probability=cnn_dropout_p)
        # Attention over the output features of the ConvolutionalNet.
        # Input: [bsz, 1, decoder_hidden_size], [bsz, image_width * image_width, cnn_hidden_num_channels * 3]
        # Output: [bsz, 1, decoder_hidden_size], [bsz, 1, image_width * image_width]
        self.visual_attention = Attention(key_size=cnn_hidden_num_channels * 3, query_size=hidden_size,
                                          hidden_size=hidden_size)

        self.textual_key_layer = nn.Sequential(nn.Linear(encoder_hidden_size, hidden_size, bias=False),
                                               nn.Tanh())
        self.auxiliary_task = auxiliary_task
        if auxiliary_task:
            self.auxiliary_loss_criterion = nn.NLLLoss()

        # Input: [batch_size, max_input_length]
        # Output: [batch_size, hidden_size], [batch_size, max_input_length, hidden_size]
        self.encoder = EncoderRNN(input_size=input_vocabulary_size,
                                  embedding_dim=embedding_dimension,
                                  rnn_input_size=embedding_dimension,
                                  hidden_size=encoder_hidden_size, num_layers=num_encoder_layers,
                                  dropout_probability=encoder_dropout_p, bidirectional=encoder_bidirectional,
                                  padding_idx=input_padding_idx)

        self.target_eos_idx = target_eos_idx
        self.target_pad_idx = target_pad_idx
        self.tanh = nn.Tanh()
        self.output_directory = output_directory

    def encode_input(self, commands_input: torch.LongTensor, commands_lengths: List[int],
                     situations_input: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Pass the input commands through an RNN encoder and the situation input through a CNN encoder."""
        encoded_image = self.situation_encoder(situations_input)
        hidden, encoder_outputs = self.encoder(commands_input, commands_lengths)
        return {"encoded_situations": encoded_image, "encoded_commands": encoder_outputs, "hidden_states": hidden}

    def summarize_input(self, encoder_hidden_states: torch.Tensor, input_lengths: List[int],
                        encoded_situations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """One decoding step based on the previous hidden state of the decoder and the previous target token."""
        batch_size, image_num_memory, _ = encoded_situations.size()
        situation_lengths = [image_num_memory for _ in range(batch_size)]
        # input_length_idx = [int(input_length - 1) for input_length in input_lengths]
        # Attention over input command and world state
        projected_keys_visual = self.visual_attention.key_layer(
            encoded_situations)  # [bsz, situation_length, hidden_dim]
        projected_keys_textual = self.textual_key_layer(
            encoder_hidden_states)  # [bsz, hidden_dim]
        # queries = projected_keys_textual[torch.tensor(input_length_idx), torch.arange(projected_keys_textual.size(1))]
        queries = projected_keys_textual
        context_situation, attention_weights_situations = self.visual_attention(
            queries=queries.unsqueeze(1), projected_keys=projected_keys_visual,
            values=projected_keys_visual, memory_lengths=situation_lengths)
        context_situation = self.tanh(context_situation)
        output = torch.cat([queries, context_situation.squeeze(1)], dim=-1)
        return output, attention_weights_situations


class ActorCritic(nn.Module):
    def __init__(self, representation_model, action_dim, encoder_hidden_size, sos_idx, pi_lr: float,
                 betas=Tuple[float, float]):
        super(ActorCritic, self).__init__()

        self.representation_model = representation_model
        self.sos_idx = sos_idx

        # Takes concatenation of the visual and language repr from ModelRL and projects to action dim.
        # Input: [bsz, encoder_hidden_size*2]
        # Output: [bsz, action_dim]
        self.action_layer = nn.Linear(encoder_hidden_size * 2, action_dim)

        # Takes concatenation of visual and language repr from ModelRL and projects to a value estimate for that state.
        # Input: [bsz, encoder_hidden_size*2]
        # Output: [bsz, 1]
        self.value_layer = nn.Linear(encoder_hidden_size * 2, 1)

        self.trained_iterations = 0
        self.best_iteration = 0
        self.best_average_reward = 0

        # log_parameters(self)
        # trainable_parameters = [parameter for parameter in self.parameters() if parameter.requires_grad]
        #
        # self.optimizer = Adam(trainable_parameters, lr=pi_lr, betas=betas)

    def forward(self):
        raise NotImplementedError

    def interact_with_policy(self, command_observations: torch.tensor,
                             command_lengths: List[int],
                             world_observations: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        """Encode input state."""
        # Encode the input.
        representations = self.representation_model.encode_input(commands_input=command_observations,
                                                                 commands_lengths=command_lengths,
                                                                 situations_input=world_observations)

        # Summarize language and vision command to shared representation with attention over image.
        (output, attention_weights_situation) = self.representation_model.summarize_input(
            encoder_hidden_states=representations["hidden_states"],
            input_lengths=command_lengths, encoded_situations=representations["encoded_situations"])
        return output, attention_weights_situation

    def pi(self, command_observations: torch.tensor, command_lengths: List[int],
           world_observations: torch.tensor, actions=torch.tensor([])) -> Tuple[torch.tensor, Any]:
        """
        Encodes the policy pi(a | s). Returns a distribution over actions given an input state and optionally
        also the log probabilities for a passed set of actions. Records computations for gradient descent.

        :param command_observations: [bsz, max_command_length], batch of command observations
        :param command_lengths: [bsz], a list containing the lengths of each command in the batch
        :param world_observations: [bsz, grid_size, grid_size, num_channels], batch of world state observations
        :param actions: [bsz], batch of actions
        :returns: a batch of policies (distribution over actions) and optionally a batch of log_probs for the actions
        """
        # Encode the input state.
        observation_representation, situation_attention_weights = self.interact_with_policy(command_observations,
                                                                                            command_lengths,
                                                                                            world_observations)

        # Get distribution over actions.
        action_representation = self.action_layer(observation_representation)
        log_probs = F.log_softmax(action_representation, dim=-1).squeeze()
        dist = Categorical(logits=log_probs)
        if not len(actions):
            return dist, None
        else:
            return dist, log_probs[torch.arange(log_probs.size(0)), actions.squeeze()]

    def pi_and_v(self, command_observations: torch.tensor, command_lengths: List[int],
                 world_observations: torch.tensor, actions=torch.tensor([])) -> Tuple[torch.tensor, torch.tensor, Any]:
        """
        Encodes the policy pi(a | s). Returns a distribution over actions given an input state and optionally
        also the log probabilities for a passed set of actions. Records computations for gradient descent.

        :param command_observations: [bsz, max_command_length], batch of command observations
        :param command_lengths: [bsz], a list containing the lengths of each command in the batch
        :param world_observations: [bsz, grid_size, grid_size, num_channels], batch of world state observations
        :param actions: [bsz], batch of actions
        :returns: a batch of policies (distribution over actions) and optionally a batch of log_probs for the actions
        """
        # Encode the input state.
        observation_representation, situation_attention_weights = self.interact_with_policy(command_observations,
                                                                                            command_lengths,
                                                                                            world_observations)

        # Get distribution over actions.
        action_representation = self.action_layer(observation_representation)
        log_probs = F.log_softmax(action_representation, dim=-1).squeeze()
        dist = Categorical(logits=log_probs)

        # Get the value estimate from the hidden state
        state_value = self.value_layer(observation_representation)

        if not len(actions):
            return dist, state_value, None
        else:
            return dist, state_value, log_probs[torch.arange(log_probs.size(0)), actions.squeeze()]

    def act(self, command_observations: torch.tensor, command_lengths: List[int],
            world_observations: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        """
        Returns set of actions chosen by the current policy. Does't record calculations for gradient descent.

        :param command_observations: [bsz, max_command_length], batch of command observations
        :param command_lengths: [bsz], a list containing the lengths of each command in the batch
        :param world_observations: [bsz, grid_size, grid_size, num_channels], batch of world state observations
        """
        with torch.no_grad():
            observation_representation, situation_attention_weights = self.interact_with_policy(
                command_observations, command_lengths, world_observations)
            action_representation = self.action_layer(observation_representation)
            log_probs = F.log_softmax(action_representation, dim=-1).squeeze()

            dist = Categorical(logits=log_probs)
            _, actions = dist.logits.max(dim=-1)
        return actions, situation_attention_weights

    def v(self, command_observations: torch.tensor, command_lengths: List[int],
          world_observations: torch.tensor) -> torch.tensor:
        """
        Encodes the value function V(s). Returns the values for a batch of observations.
        The value of a state is the expected future reward from that state.

        :param command_observations: [bsz, max_command_length], batch of command observations
        :param command_lengths: [bsz], a list containing the lengths of each command in the batch
        :param world_observations: [bsz, grid_size, grid_size, num_channels], batch of world state observations
        :returns: a batch of values for the observations
        """
        observation_representation, situation_attention_weights = self.interact_with_policy(command_observations,
                                                                                            command_lengths,
                                                                                            world_observations)
        # Get the value estimate from the hidden state
        state_value = self.value_layer(observation_representation)
        return state_value

    def step(self, command_observations: torch.tensor,
             command_lengths: List[int],
             world_observations: torch.tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns the actions and values for an input state. Doesn't record computations for gradient descent.

        :param command_observations: [bsz, max_command_length], batch of command observations
        :param command_lengths: [bsz], a list containing the lengths of each command in the batch
        :param world_observations: [bsz, grid_size, grid_size, num_channels], batch of world state observations

        :return: tuple of actions [bsz, action_dim], values [bsz,], logprobs for actions [bsz,]
        """
        with torch.no_grad():
            observation_representation, situation_attention_weights = self.interact_with_policy(
                command_observations, command_lengths, world_observations)
            action_representation = self.action_layer(observation_representation)
            log_probs = F.log_softmax(action_representation, dim=-1)

            dist = Categorical(logits=log_probs)
            action = dist.sample()

            # Get the value estimate from the hidden state
            state_value = self.value_layer(observation_representation)

        return action, state_value, log_probs.squeeze(), situation_attention_weights

    def update_state(self, is_best: bool, average_reward=None) -> {}:
        self.trained_iterations += 1
        if is_best:
            self.best_average_reward = average_reward
            self.best_iteration = self.trained_iterations

    def load_model(self, path_to_checkpoint: str) -> dict:
        checkpoint = torch.load(path_to_checkpoint)
        self.trained_iterations = checkpoint["iteration"]
        self.best_iteration = checkpoint["best_iteration"]
        self.load_state_dict(checkpoint["state_dict"])
        self.best_average_reward = checkpoint["best_average_reward"]
        return checkpoint["optimizer_state_dict"]

    def get_current_state(self):
        return {
            "iteration": self.trained_iterations,
            "state_dict": self.state_dict(),
            "best_iteration": self.best_iteration,
            "best_average_reward": self.best_average_reward
        }

    def save_checkpoint(self, file_name: str, is_best: bool, optimizer_state_dict: dict) -> str:
        """

        :param file_name: filename to save checkpoint in.
        :param is_best: boolean describing whether or not the current state is the best the model has ever been.
        :param optimizer_state_dict: state of the optimizer.
        :return: str to path where the model is saved.
        """
        path = os.path.join(self.output_directory, file_name)
        state = self.get_current_state()
        state["optimizer_state_dict"] = optimizer_state_dict
        torch.save(state, path)
        if is_best:
            best_path = os.path.join(self.output_directory, 'model_best.pth.tar')
            shutil.copyfile(path, best_path)
        return path


class ActorCriticPolicy(nn.Module):
    def __init__(self, observation_space, action_space, lr_schedule, use_sde,
                 representation_model, action_dim, encoder_hidden_size, sos_idx, pi_lr: float,
                 betas=Tuple[float, float]):
        super(ActorCriticPolicy, self).__init__()

        self.observation_space = observation_space
        self.action_space = action_space
        self.lr_schedule = lr_schedule,
        self.use_sde = use_sde
        self.representation_model = representation_model
        self.sos_idx = sos_idx

        # Takes concatenation of the visual and language repr from ModelRL and projects to action dim.
        # Input: [bsz, encoder_hidden_size*2]
        # Output: [bsz, action_dim]
        self.action_layer = nn.Linear(encoder_hidden_size * 2, action_dim)

        # Takes concatenation of visual and language repr from ModelRL and projects to a value estimate for that state.
        # Input: [bsz, encoder_hidden_size*2]
        # Output: [bsz, 1]
        self.value_layer = nn.Linear(encoder_hidden_size * 2, 1)

        self.trained_iterations = 0
        self.best_iteration = 0
        self.best_average_reward = 0

        log_parameters(self)
        trainable_parameters = [parameter for parameter in self.parameters() if parameter.requires_grad]

        self.optimizer = Adam(trainable_parameters, lr=pi_lr, betas=betas)

    def forward(self, observations):
        command_observations, command_lengths, world_observations = self.unpack_observation(observations)
        return self.pi_and_v(command_observations, command_lengths, world_observations)

    def evaluate_actions(self, observations, actions):
        command_observations, command_lengths, world_observations = self.unpack_observation(observations)
        return self.pi_and_v(command_observations, command_lengths, world_observations, actions=actions)

    def unpack_observation(self, observations: torch.tensor) -> Tuple[torch.tensor, List[int], torch.tensor]:
        if len(observations.shape) == 3:
            observations = np.expand_dims(observations, axis=0)
        command_observations = observations[:, -1, 0, :]
        command_lengths = list(int(l) for l in (torch.tensor(command_observations) == 2).nonzero()[:, -1] + 1)
        world_observations = observations[:, :-1, :, :]
        return command_observations.to(dtype=torch.long), command_lengths, world_observations.to(dtype=torch.float32)

    def interact_with_policy(self, command_observations: torch.tensor,
                             command_lengths: List[int],
                             world_observations: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        """Encode input state."""
        # Encode the input.
        representations = self.representation_model.encode_input(commands_input=command_observations,
                                                                 commands_lengths=command_lengths,
                                                                 situations_input=world_observations)

        # Summarize language and vision command to shared representation with attention over image.
        (output, attention_weights_situation) = self.representation_model.summarize_input(
            encoder_hidden_states=representations["hidden_states"],
            input_lengths=command_lengths, encoded_situations=representations["encoded_situations"])
        return output, attention_weights_situation

    def pi(self, command_observations: torch.tensor, command_lengths: List[int],
           world_observations: torch.tensor, actions=torch.tensor([])) -> Tuple[torch.tensor, Any]:
        """
        Encodes the policy pi(a | s). Returns a distribution over actions given an input state and optionally
        also the log probabilities for a passed set of actions. Records computations for gradient descent.

        :param command_observations: [bsz, max_command_length], batch of command observations
        :param command_lengths: [bsz], a list containing the lengths of each command in the batch
        :param world_observations: [bsz, grid_size, grid_size, num_channels], batch of world state observations
        :param actions: [bsz], batch of actions
        :returns: a batch of policies (distribution over actions) and optionally a batch of log_probs for the actions
        """
        # Encode the input state.
        observation_representation, situation_attention_weights = self.interact_with_policy(command_observations,
                                                                                            command_lengths,
                                                                                            world_observations)

        # Get distribution over actions.
        action_representation = self.action_layer(observation_representation)
        log_probs = F.log_softmax(action_representation, dim=-1).squeeze()
        dist = Categorical(logits=log_probs)
        if not len(actions):
            return dist, None
        else:
            return dist, log_probs[torch.arange(log_probs.size(0)), actions.squeeze()]

    def pi_and_v(self, command_observations: torch.tensor, command_lengths: List[int],
                 world_observations: torch.tensor, actions=torch.tensor([])) -> Tuple[torch.tensor, torch.tensor, Any]:
        """
        Encodes the policy pi(a | s). Returns a distribution over actions given an input state and optionally
        also the log probabilities for a passed set of actions. Records computations for gradient descent.

        :param command_observations: [bsz, max_command_length], batch of command observations
        :param command_lengths: [bsz], a list containing the lengths of each command in the batch
        :param world_observations: [bsz, grid_size, grid_size, num_channels], batch of world state observations
        :param actions: [bsz], batch of actions
        :returns: a batch of policies (distribution over actions) and optionally a batch of log_probs for the actions
        """
        # Encode the input state.
        observation_representation, situation_attention_weights = self.interact_with_policy(command_observations,
                                                                                            command_lengths,
                                                                                            world_observations)

        # Get distribution over actions.
        action_representation = self.action_layer(observation_representation)
        log_probs = F.log_softmax(action_representation, dim=-1).squeeze()
        dist = Categorical(logits=log_probs)

        # Get the value estimate from the hidden state
        state_value = self.value_layer(observation_representation)

        if not len(actions):
            _, actions = dist.logits.max(dim=-1)
            if len(log_probs.shape) == 1:
                log_probs = log_probs.unsqueeze(0)
            if not len(actions.shape):
                actions = actions.unsqueeze(dim=0)
            return actions, state_value, log_probs[torch.arange(log_probs.size(0)), actions.squeeze()]
        else:
            return state_value, log_probs[torch.arange(log_probs.size(0)), actions.squeeze()], dist.entropy()

    def predict(self, observations, state=None, mask=None, deterministic=False):
        command_observations, command_lengths, world_observations = self.unpack_observation(observations)
        return self.act(command_observations, command_lengths, world_observations)

    def act(self, command_observations: torch.tensor, command_lengths: List[int],
            world_observations: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        """
        Returns set of actions chosen by the current policy. Does't record calculations for gradient descent.

        :param command_observations: [bsz, max_command_length], batch of command observations
        :param command_lengths: [bsz], a list containing the lengths of each command in the batch
        :param world_observations: [bsz, grid_size, grid_size, num_channels], batch of world state observations
        """
        with torch.no_grad():
            observation_representation, situation_attention_weights = self.interact_with_policy(
                command_observations, command_lengths, world_observations)
            action_representation = self.action_layer(observation_representation)
            log_probs = F.log_softmax(action_representation, dim=-1).squeeze()

            dist = Categorical(logits=log_probs)
            _, actions = dist.logits.max(dim=-1)
        return actions, situation_attention_weights

    def v(self, command_observations: torch.tensor, command_lengths: List[int],
          world_observations: torch.tensor) -> torch.tensor:
        """
        Encodes the value function V(s). Returns the values for a batch of observations.
        The value of a state is the expected future reward from that state.

        :param command_observations: [bsz, max_command_length], batch of command observations
        :param command_lengths: [bsz], a list containing the lengths of each command in the batch
        :param world_observations: [bsz, grid_size, grid_size, num_channels], batch of world state observations
        :returns: a batch of values for the observations
        """
        observation_representation, situation_attention_weights = self.interact_with_policy(command_observations,
                                                                                            command_lengths,
                                                                                            world_observations)
        # Get the value estimate from the hidden state
        state_value = self.value_layer(observation_representation)
        return state_value

    def step(self, command_observations: torch.tensor,
             command_lengths: List[int],
             world_observations: torch.tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns the actions and values for an input state. Doesn't record computations for gradient descent.

        :param command_observations: [bsz, max_command_length], batch of command observations
        :param command_lengths: [bsz], a list containing the lengths of each command in the batch
        :param world_observations: [bsz, grid_size, grid_size, num_channels], batch of world state observations

        :return: tuple of actions [bsz, action_dim], values [bsz,], logprobs for actions [bsz,]
        """
        with torch.no_grad():
            observation_representation, situation_attention_weights = self.interact_with_policy(
                command_observations, command_lengths, world_observations)
            action_representation = self.action_layer(observation_representation)
            log_probs = F.log_softmax(action_representation, dim=-1)

            dist = Categorical(logits=log_probs)
            action = dist.sample()

            # Get the value estimate from the hidden state
            state_value = self.value_layer(observation_representation)

        return action, state_value, log_probs.squeeze(), situation_attention_weights

    def update_state(self, is_best: bool, average_reward=None) -> {}:
        self.trained_iterations += 1
        if is_best:
            self.best_average_reward = average_reward
            self.best_iteration = self.trained_iterations

    def load_model(self, path_to_checkpoint: str) -> dict:
        checkpoint = torch.load(path_to_checkpoint)
        self.trained_iterations = checkpoint["iteration"]
        self.best_iteration = checkpoint["best_iteration"]
        self.load_state_dict(checkpoint["state_dict"])
        self.best_average_reward = checkpoint["best_average_reward"]
        return checkpoint["optimizer_state_dict"]

    def get_current_state(self):
        return {
            "iteration": self.trained_iterations,
            "state_dict": self.state_dict(),
            "best_iteration": self.best_iteration,
            "best_average_reward": self.best_average_reward
        }

    def save_checkpoint(self, file_name: str, is_best: bool, optimizer_state_dict: dict) -> str:
        """

        :param file_name: filename to save checkpoint in.
        :param is_best: boolean describing whether or not the current state is the best the model has ever been.
        :param optimizer_state_dict: state of the optimizer.
        :return: str to path where the model is saved.
        """
        path = os.path.join(self.output_directory, file_name)
        state = self.get_current_state()
        state["optimizer_state_dict"] = optimizer_state_dict
        torch.save(state, path)
        if is_best:
            best_path = os.path.join(self.output_directory, 'model_best.pth.tar')
            shutil.copyfile(path, best_path)
        return path