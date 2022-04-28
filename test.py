#!/usr/bin/env python3

import os
import cv2 as cv
import numpy as np
import yaml
import torch as th
import gym
import random
import datetime as dt
import pandas as pd

import rl
import gym_search

from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter
from time import process_time
from tqdm import tqdm
from gym_search.utils import travel_dist


KEY_ESC = 27
KEY_RET = 13
WINDOW_SIZE = (640, 640)


def spl(success, shortest, taken):
    # https://arxiv.org/pdf/1807.06757.pdf
    return success*shortest/np.maximum(shortest, taken)


def parse_hparams(s):
    if os.path.exists(s):
        with open(s) as f:
            return yaml.load(f, Loader=yaml.FullLoader)

    return yaml.safe_load(s)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("environment", type=str)
    parser.add_argument("--env-kwargs", type=parse_hparams, default={})
    parser.add_argument("--agent", type=str)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--name", type=str, default=dt.datetime.now().isoformat())
    parser.add_argument("--model", type=str)
    parser.add_argument("--delay", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--observe", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--hidden", action="store_true")

    args = parser.parse_args()

    wrappers = [
        gym.wrappers.RecordEpisodeStatistics,
        gym_search.wrappers.ResizeImage,
        gym_search.wrappers.LastAction
    ]

    env = gym.make(args.environment, **args.env_kwargs)
    for wrapper in wrappers:
        env = wrapper(env)

    agent = None
    device = th.device(args.device)
    writer = SummaryWriter(f"logs/test/{args.name}")
    df = pd.DataFrame()
    infos = []

    if args.name is None:
        args.name = dt.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        th.manual_seed(args.seed)
    
    if args.deterministic: 
        th.use_deterministic_algorithms(True)

    if args.model:
        print(f"loading {args.model}")
        agent = th.load(args.model).to(device)
        agent.eval()

    if args.record:
        env = gym.wrappers.RecordVideo(env, "videos", episode_trigger=lambda _: True, name_prefix=args.name)

    if not args.hidden:
        cv.namedWindow(args.environment, cv.WINDOW_AUTOSIZE)

    env.test()

    for ep in tqdm(range(args.episodes)):

        done = False
        seed = args.seed if ep == 0 else None
        obs = env.reset(seed=seed)

        if agent is not None:
            state = [s.to(device) for s in agent.initial(1)]

        while not done:
            key = None
            img = None
            act = None

            if not args.hidden:
                if args.observe:
                    img = obs["image"]
                else:
                    img = env.render(mode="rgb_array")

                img = cv.resize(img, WINDOW_SIZE, interpolation=cv.INTER_AREA)
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

                cv.imshow(args.environment, img)
                key = cv.waitKey(args.delay)

            if agent is None:
                if args.agent == "random":
                    act = env.get_random_action()
                elif args.agent == "greedy":
                    act = env.get_greedy_action()
                else:
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

        length = float(len(info["path"]))
        success = float(info["success"])
        shortest = float(travel_dist(info["targets"] + [info["initial"]]) + len(info["targets"]))

        writer.add_scalar("metric/length", length, ep)
        writer.add_scalar("metric/shortest", shortest, ep)
        writer.add_scalar("metric/spl", spl(success, shortest, length), ep)

        infos.append(info)

    success = np.array([info["success"] for info in infos], dtype=float)
    shortest = np.array([travel_dist(info["targets"] + [info["initial"]]) + len(info["targets"]) for info in infos], dtype=float)
    taken = np.array([len(info["path"]) for info in infos], dtype=float)

    writer.add_histogram("metric/shortest", shortest)
    writer.add_histogram("metric/taken", taken)
    writer.add_histogram("metric/spl", spl(success, shortest, taken))

    print("mean length:", np.mean(taken))
    print("mean spl:", np.mean(spl(success, shortest, taken)))