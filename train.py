#!/usr/bin/env python3

import argparse
import datetime

import gym
import gym_ptz

from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback


parser = argparse.ArgumentParser()
parser.add_argument("env", type=str)
parser.add_argument("-p", "--policy", type=str, choices=["MlpPolicy", "CnnPolicy", "MultiInputPolicy"], default="MlpPolicy")
parser.add_argument("-l", "--logs", type=str, default="logs")

args = parser.parse_args()
env = make_vec_env(args.env, n_envs=8)
model = PPO(args.policy, env, verbose=1, tensorboard_log=args.logs, ent_coef=0.01)
name = f"ppo-{args.policy.lower()}-{datetime.datetime.now().isoformat()}"

checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=args.logs, name_prefix='ckpt')

model.learn(total_timesteps=int(10e6), tb_log_name=name, callback=checkpoint_callback)
model.save(name)

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
