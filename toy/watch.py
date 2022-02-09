#!/usr/bin/env python3

import gym

import torch as th
import torch.nn as nn

from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv

import envs


env = DummyVecEnv([lambda: gym.make("Coverage-v0", width=400, height=400, radius=20)])
model = PPO.load("ppo", env=env)

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
