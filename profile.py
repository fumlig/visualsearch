#!/usr/bin/env python3

import argparse
import time
import gym
import gym_search

from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("env", type=str)
parser.add_argument("-s", "--steps", type=int, default=100000)

args = parser.parse_args()

env = gym.make(args.env)

obs = env.reset()

start = time.time()

for _ in tqdm(range(args.steps)):
    action = env.action_space.sample()    
    obs, reward, done, _info = env.step(action)

    if done:
        obs = env.reset()

end = time.time()

seconds = end - start
frames = args.steps

print("fps:", frames/seconds)