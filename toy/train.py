#!/usr/bin/env python3

import gym

import torch as th
import torch.nn as nn

from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env


from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import envs


ENV = "Coverage-v"


# Parallel environments
env = make_vec_env("Coverage-v0", n_envs=4, env_kwargs={"width": 100, "height": 100})


policy_kwargs = dict(
    activation_fn=th.nn.ReLU,
    net_arch=[dict(pi=[32, 32], vf=[32, 32])],
    features_extractor_kwargs=dict(features_dim=128),
)

model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="logs", ent_coef=0.001, policy_kwargs=policy_kwargs)
model.learn(total_timesteps=1000000, tb_log_name="ppo")

obs = env.reset()

while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()


# ppo, ent_coef=0.01:   ~14
# ppo, ent_coef=0.001:  ~18
# ppo, ent_coef=0.0:    