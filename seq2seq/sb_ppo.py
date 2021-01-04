import numpy as np
import random
import torch
from stable_baselines3 import PPO
from stable_baselines.common.env_checker import check_env

from seq2seq.groundedscanenv import GroundedScanEnvironment
from seq2seq.rl_model import ModelRL, ActorCriticPolicy


def train_ppo(training_env: GroundedScanEnvironment, simple_situation_representation: bool, ppo_log_interval: int,
              max_episode_length: int, timesteps_per_epoch: int, num_epochs: int,
              pi_lr: float, adam_beta_1: float, training_batch_size: int,
              adam_beta_2: float, hidden_size: int, embedding_dimension: int, encoder_hidden_size: int,
              num_encoder_layers: int, encoder_bidirectional: bool, num_decoder_layers: int, cnn_dropout_p: float,
              decoder_dropout_p: float, train_pi_iters: int, cnn_kernel_size: int, cnn_hidden_num_channels: int,
              encoder_dropout_p: float, auxiliary_task: bool, conditional_attention: bool, attention_type: str,
              evaluate_every_epoch: int, ppo_log_every: int, visualize_trajectory_every: int, output_directory: str,
              gamma=0.99, gae_lambda=0.97, target_kl=0.01, clip_ratio=0.2, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
              seed=1234, **kwargs):
    # Random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    action_dim = training_env.action_space.n
    betas = (adam_beta_1, adam_beta_2)
    check_env(training_env)

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
    actor_critic = ActorCriticPolicy
    policy_kwargs = {"action_dim": action_dim, "encoder_hidden_size": hidden_size,
                     "sos_idx": training_env.target_vocabulary.sos_idx, "pi_lr": pi_lr,
                     "betas": betas, "representation_model": representation_model}

    # initial_obs = training_env.reset()
    # next_obs = training_env.step(1)
    model = PPO(actor_critic, training_env, learning_rate=pi_lr, n_steps=timesteps_per_epoch,
                batch_size=training_batch_size, n_epochs=10, gamma=gamma, gae_lambda=gae_lambda, clip_range=clip_ratio,
                ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=max_grad_norm, target_kl=target_kl, seed=seed,
                verbose=1, policy_kwargs=policy_kwargs, use_sde=False)
    model.learn(total_timesteps=40000)
    print()
    obs = training_env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = training_env.step(action.item())
        training_env.render()