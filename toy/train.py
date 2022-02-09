#!/usr/bin/env python3

import gym

import torch as th
import torch.nn as nn

from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env

import envs

# probably easier for it to learn if it only sees tiles around itself
# more fixed mapping

env = make_vec_env("Coverage-v0", n_envs=4, env_kwargs={"width": 400, "height": 400, "radius": 20})
model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="logs", ent_coef=0.01)

model.learn(total_timesteps=1000000, tb_log_name="ppo")

model.save("ppo")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
