#!/usr/bin/env python3

from argparse import ArgumentParser
from collections import defaultdict
from time import process_time

import cv2 as cv
import numpy as np
import torch as th
import gym
import gym_search
import random
import datetime as dt

from gym_search.utils import travel_dist

import rl
import gym_search


KEY_ESC = 27
KEY_RET = 13
WINDOW_SIZE = (640, 640)


def spl(success, shortest_path, taken_path):
    # https://arxiv.org/pdf/1807.06757.pdf
    return success*shortest_path/max(shortest_path, taken_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("environment", type=str)
    parser.add_argument("--agent", type=str)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--name", type=str, default=dt.datetime.now().isoformat())
    parser.add_argument("--model", type=str)
    parser.add_argument("--delay", type=int, default=1)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--observe", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--record", action="store_true")

    args = parser.parse_args()

    wrappers = [
        gym.wrappers.RecordEpisodeStatistics,
        gym_search.wrappers.ResizeImage,
    ]

    env = gym.make(args.environment)
    for wrapper in wrappers:
        env = wrapper(env)

    agent = None
    device = th.device("cuda" if th.cuda.is_available() and not args.cpu else "cpu")
    infos = []

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        th.manual_seed(args.seed)
        env.seed(args.seed)

    if args.agent is not None:
        agent = rl.agent(args.agent)(env)

    if args.model:
        print(f"loading {args.model}")
        agent = th.load(args.model).to(device)
        agent.eval()

    if args.record:
        env = gym.wrappers.RecordVideo(env, "videos", episode_trigger=lambda _: True, name_prefix=args.name)

    cv.namedWindow(args.environment, cv.WINDOW_AUTOSIZE)

    for ep in range(args.episodes):

        done = False
        obs = env.reset()

        if agent is not None:
            state = [s.to(device) for s in agent.initial(1)]

        while not done:
            if args.observe:
                img = obs["image"]
            else:
                img = env.render(mode="rgb_array")

            img = cv.resize(img, WINDOW_SIZE, interpolation=cv.INTER_AREA)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

            cv.imshow(args.environment, img)	

            key = cv.waitKey(args.delay)

            if agent is None:
                act = env.get_keys_to_action().get((key,), 0)
            else:
                with th.no_grad():
                    obs = {key: th.tensor(sub_obs).float().unsqueeze(0).to(device) for key, sub_obs in obs.items()}
                    act, state = agent.predict(obs, state, done=th.tensor(done).float().unsqueeze(0).to(device), deterministic=args.deterministic)

            
            step_begin = process_time()
            obs, rew, done, info = env.step(act)
            step_end = process_time()

            if key == KEY_RET:
                done = True

            if key == KEY_ESC:
                exit(0)

            if args.verbose:
                print(
                    "action:", env.get_action_meanings()[act],
                    "observation:", obs,
                    "reward:", rew,
                    "info:", info,
                    "fps:", 1.0/(step_end - step_begin)
                )

            if done:
                infos.append(info)

        # todo: spl
        print(infos)