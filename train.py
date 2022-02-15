#!/usr/bin/env python3

import gym
import gym_ptz

from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback


env = make_vec_env("Toy-v0", n_envs=8)
model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log="logs", ent_coef=0.01)

checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='logs', name_prefix='ckpt')

model.learn(total_timesteps=int(25e6), tb_log_name="ppo", callback=checkpoint_callback)

model.save("ppo")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
