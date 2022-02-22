#!/usr/bin/env python3

import gym
import argparse
import torch as th
import torch.nn as nn

from ppo import Agent

import gym_search


parser = argparse.ArgumentParser()
parser.add_argument("env", type=str)
parser.add_argument("model", type=str)
parser.add_argument("--num-envs", type=int, default=8)
args = parser.parse_args()


#envs = gym.vector.make(args.env, args.num_envs, asynchronous=False, wrappers=[gym.wrappers.FlattenObservation])
env = gym.wrappers.FlattenObservation(gym.make(args.env))
device = th.device("cuda" if th.cuda.is_available() else "cpu")
agent = th.load(args.model).to(device)
obs = env.reset()

agent.eval()

while True:
    with th.no_grad():
        pi, vf = agent(th.Tensor(obs).to(device))
        act = pi.sample().item()
    
    obs, rew, done, info = env.step(act)
    env.render()
