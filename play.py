#!/usr/bin/env python3


from argparse import ArgumentParser
from collections import defaultdict

import cv2 as cv
import numpy as np
import torch as th
import gym
import gym_search

from ppo import Agent


KEY_ESC = 27
KEY_RET = 13
WINDOW_SIZE = (640, 640)


parser = ArgumentParser()
parser.add_argument("env", type=str)
parser.add_argument("-m", "--model", type=str)
parser.add_argument("-d", "--delay", type=int, default=1)
parser.add_argument("-o", "--observe", action="store_true")
parser.add_argument("-v", "--verbose", action="store_true")

args = parser.parse_args()

env = gym.wrappers.FlattenObservation(gym.make(args.env))
device = th.device("cuda" if th.cuda.is_available() else "cpu")
agent = None
obs = env.reset()

stats = defaultdict(int)

if args.model:
    agent = th.load(args.model).to(device)
    agent.eval()

cv.namedWindow(args.env, cv.WINDOW_AUTOSIZE)

while cv.getWindowProperty(args.env, cv.WND_PROP_VISIBLE) > 0:
    img = env.render(mode="rgb_array", observe=args.observe)

    if args.observe:
        img = (img*255).astype(dtype=np.uint8)

    img = cv.resize(img, WINDOW_SIZE, interpolation=cv.INTER_NEAREST)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    cv.imshow(args.env, img)

    key = cv.waitKey(args.delay)

    if key == KEY_ESC:
        break

    if agent is not None:
        with th.no_grad():
            pi, vf = agent(th.Tensor(obs).to(device))
            act = pi.sample().item()
    else:
        act = {
            ord(" "): env.Action.TRIGGER,
            ord("w"): env.Action.NORTH,
            ord("d"): env.Action.EAST,
            ord("s"): env.Action.SOUTH,
            ord("a"): env.Action.WEST,
        }.get(key, env.Action.NONE)
    
    obs, rew, done, info = env.step(act)

    stats["return"] += rew
    stats["triggers"] += act == env.Action.TRIGGER

    if args.verbose:
        print("reward:", rew, "action:", env.get_action_meanings()[act])

    if done or key == KEY_RET:
        print(stats)
        obs = env.reset()
        stats.clear()
