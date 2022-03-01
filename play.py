#!/usr/bin/env python3


from argparse import ArgumentParser
from collections import defaultdict

import cv2 as cv
import numpy as np
import torch as th
import gym
import gym_search

from ppo import Agent
from agents.random import RandomAgent

KEY_ESC = 27
KEY_RET = 13
WINDOW_SIZE = (640, 640)


parser = ArgumentParser()
parser.add_argument("env", type=str)
parser.add_argument("--agent", type=str)
parser.add_argument("--delay", type=int, default=1)
parser.add_argument("--observe", action="store_true")
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--episodes", type=int, default=100)

args = parser.parse_args()

env = gym.wrappers.FlattenObservation(gym.make(args.env))
device = th.device("cuda" if th.cuda.is_available() else "cpu")
stats = [defaultdict(int) for _ in range(args.episodes)]

if args.agent is None:
    agent = None
elif args.agent == "random":
    agent = RandomAgent(env)
else:
    agent = th.load(args.agent).to(device)
    agent.eval()

cv.namedWindow(args.env, cv.WINDOW_AUTOSIZE)

for ep in range(args.episodes):
    
    done = False
    obs = env.reset()

    while not done:
        img = env.render(mode="rgb_array", observe=args.observe)
        img = cv.resize(img, WINDOW_SIZE, interpolation=cv.INTER_NEAREST)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        cv.imshow(args.env, img)

        key = cv.waitKey(args.delay)

        if agent is None:
            act = env.get_keys_to_action().get((key,), 0)
        else:
            with th.no_grad():
                act = agent.predict(th.tensor(obs, dtype=th.float).to(device))
        
        obs, rew, done, info = env.step(act)

        if key == KEY_RET:
            done = True

        if key == KEY_ESC:
            exit(0)

        stats[ep]["steps"] += 1
        stats[ep]["return"] += rew
        stats[ep]["triggers"] += act == env.Action.TRIGGER

        if args.verbose:
            print("action:", env.get_action_meanings()[act], "reward:", rew)

    if done:
        print(", ".join([f"{key}: {value}" for key, value in stats[ep].items()]))


print(
    "average return:", sum([ep["return"] for ep in stats])/len(stats),
    "average length:", sum([ep["steps"] for ep in stats])/len(stats),
    "average triggers:", sum([ep["triggers"] for ep in stats])/len(stats),
)