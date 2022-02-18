#!/usr/bin/env python3

import re
import argparse
import datetime
import yaml

import torch as th
import torch.nn as nn
import gym
import gym_search

parser = argparse.ArgumentParser()
parser.add_argument("env", type=str)
parser.add_argument("algo", type=str, default="ppo")
parser.add_argument("--hparams", type=str, nargs="*")
parser.add_argument("--logs", type=str, default="logs")

def parse_hparams(hparams_arg):
    if hparams_arg is None:
        return {}
    
    hparams = {}
    
    for hparam in hparams_arg:
        key, type, value = re.split(":|=", hparam)
        hparams[key] = pydoc.locate(type)(value)

    return hparams

args = parser.parse_args()
env = gym.vector.make(args.env, num_envs=8, wrappers=[gym.wrappers.FlattenObservation], asynchronous=False)

name = f"{args.env.lower()}-{args.algo.lower()}-{datetime.datetime.now().isoformat()}"

hparams = parse_hparams(args.hparams)

print(name)
print(hparams)

"""
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
"""