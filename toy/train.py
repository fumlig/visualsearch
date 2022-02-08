#!/usr/bin/env python3

import gym

import torch as th
import torch.nn as nn

from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env

import envs

# Parallel environments
env = make_vec_env("Coverage-v0", n_envs=4, env_kwargs={"width": 25, "height": 10})

model = PPO("MlpPolicy", env, verbose=1, ent_coef=0.001, gamma=1.0)
model.learn(total_timesteps=1000000)
model.save("ppo_pantiltzoom")

del model # remove to demonstrate saving and loading

model = PPO.load("ppo_pantiltzoom")

obs = env.reset()

while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
