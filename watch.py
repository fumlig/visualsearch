#!/usr/bin/env python3

import gym
import argparse
import torch as th
import torch.nn as nn

from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv

import gym_search


parser = argparse.ArgumentParser()
parser.add_argument("env", type=str)
parser.add_argument("model", type=str)
args = parser.parse_args()

env = VecVideoRecorder(DummyVecEnv([lambda: gym.make(args.env)]), "videos", record_video_trigger=lambda _: True) 
model = PPO.load(args.model, env=env)

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
