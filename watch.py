#!/usr/bin/env python3

import gym
import argparse
import torch as th
import torch.nn as nn

from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv

import envs


parser = argparse.ArgumentParser()
parser.add_argument("model", type=str)
args = parser.parse_args()

env = DummyVecEnv([lambda: gym.make("Coverage-v0")])
model = PPO.load(args.model, env=env)

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
