#!/usr/bin/env python3

from argparse import ArgumentParser
from collections import defaultdict
from time import process_time

import cv2 as cv
import numpy as np
import torch as th
import gym
import gym_search

from gym_search.utils import travel_dist
from gym_search.wrappers import ObserveVisible, ResizeImage, ObserveTime, ObserveVisible, ObserveVisited

from agents.ac import ActorCritic
from agents.random import RandomAgent

KEY_ESC = 27
KEY_RET = 13
WINDOW_SIZE = (640, 640)


parser = ArgumentParser()
parser.add_argument("env", type=str)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--agent", type=str)
parser.add_argument("--delay", type=int, default=1)
parser.add_argument("--observe", action="store_true")
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--episodes", type=int, default=1024)

args = parser.parse_args()

env = gym.make(args.env)

wrappers = [gym.wrappers.RecordEpisodeStatistics, ResizeImage]#, ObserveTime, ObserveVisible, ObserveVisited]

for wrapper in wrappers:
    env = wrapper(env)

device = th.device("cuda" if th.cuda.is_available() else "cpu")
stats = [defaultdict(int) for _ in range(args.episodes)]

if args.seed is not None:
    env.seed(args.seed)

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

    #if args.verbose:
    #    points = [env.view.pos] + [target.pos for target in env.targets]
    #    print("optimal:", travel_dist(points))

    while not done:
        if args.observe:
            img = obs["image"]
            print(obs)
        else:
            img = env.render(mode="rgb_array")

        img = cv.resize(img, WINDOW_SIZE, interpolation=cv.INTER_AREA)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        cv.imshow(args.env, img)	

        key = cv.waitKey(args.delay)

        if agent is None:
            act = env.get_keys_to_action().get((key,), 0)
        else:
            with th.no_grad():
                obs = {key: th.tensor(sub_obs).float().unsqueeze(0).to(device) for key, sub_obs in obs.items()}
                act = agent.predict(obs)
        
        if args.verbose:
            step_begin = process_time()
        
        obs, rew, done, info = env.step(act)
        
        if args.verbose:
            step_end = process_time()
            print("fps:", 1.0/(step_end - step_begin))

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
        #cv.imwrite("search.jpg", env._image(show_path=True))


print(
    "average return:", sum([ep["return"] for ep in stats])/len(stats),
    "average length:", sum([ep["steps"] for ep in stats])/len(stats),
    "average triggers:", sum([ep["triggers"] for ep in stats])/len(stats),
)