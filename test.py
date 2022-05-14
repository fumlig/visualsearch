#!/usr/bin/env python3

import os
import cv2 as cv
import csv
import numpy as np
import yaml
import torch as th
import gym
import random
import datetime as dt
import pandas as pd
import glob

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

BASELINES = ["human", "greedy", "random", "exhaustive"]


def spl_metric(success, taken, shortest):
    # https://arxiv.org/pdf/1807.06757.pdf
    return success*shortest/np.maximum(shortest, taken)


def parse_hparams(s):
    if os.path.exists(s):
        with open(s) as f:
            return yaml.load(f, Loader=yaml.FullLoader)

    return yaml.safe_load(s)


def play(episodes, env, agent="human", model=None, device=None, hidden=False, observe=False, delay=1, seed=None, deterministic=False):
    infos = []

    if not hidden:
        cv.namedWindow(env.spec.id, cv.WINDOW_AUTOSIZE)

    for ep in tqdm(range(episodes), leave=False):
        done = False
        obs = env.reset(seed=seed if ep == 0 else None)

        if model is not None:
            state = [s.to(device) for s in model.initial(1)]

        while not done:
            key = None
            img = None
            act = None

            if not hidden:
                if observe:
                    img = obs["image"]
                    print("position:", obs["position"])
                else:
                    img = env.render(mode="rgb_array")

                img = cv.resize(img, WINDOW_SIZE, interpolation=cv.INTER_AREA)
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

                cv.imshow(env.spec.id, img)
                key = cv.waitKey(delay)

            if model is not None:
                with th.no_grad():
                    obs = {key: th.tensor(sub_obs).float().unsqueeze(0).to(device) for key, sub_obs in obs.items()}
                    act, state = model.predict(obs, state, done=th.tensor(done).float().unsqueeze(0).to(device), deterministic=deterministic)
            elif agent == "human":
                act = env.get_keys_to_action().get((key,), 0)
            elif agent == "random":
                act = env.get_random_action()
            elif agent == "greedy":
                act = env.get_greedy_action()
            elif agent == "exhaustive":
                act = env.get_exhaustive_action()
            else:
                raise ValueError(f"agent must be one of {','.join(BASELINES)}")

            step_begin = process_time()
            obs, rew, done, info = env.step(act)
            step_end = process_time()

            if key == KEY_RET:
                done = True

            if key == KEY_ESC:
                exit(0)

            """
            print(
                "action:", env.get_action_meanings()[act],
                #"observation:", obs,
                "reward:", rew,
                "info:", info,
                "fps:", 1.0/(step_end - step_begin)
            )
            """

        infos.append(info)

    return infos


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("environment", type=str)
    parser.add_argument("--env-kwargs", type=parse_hparams, default={})
    parser.add_argument("--agent", type=str, choices=BASELINES, default="human")
    parser.add_argument("--models", type=str, nargs="*", default=[])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--name", type=str, default=dt.datetime.now().isoformat())
    parser.add_argument("--delay", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--observe", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--hidden", action="store_true")
    parser.add_argument("--deterministic", action="store_true")

    args = parser.parse_args()

    model_paths = []

    for path in args.models:
        model_paths += glob.glob(path)
    
    model_paths.sort(key=lambda x: os.path.getmtime(x))

    wrappers = [
        gym.wrappers.RecordEpisodeStatistics,
        gym_search.wrappers.ResizeImage,
        gym_search.wrappers.LastAction
    ]

    env = gym.make(args.environment, **args.env_kwargs)
    for wrapper in wrappers:
        env = wrapper(env)

    device = th.device(args.device)
    df = pd.DataFrame()

    if args.name is None:
        args.name = dt.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')

    if args.record:
        env = gym.wrappers.RecordVideo(env, "videos", episode_trigger=lambda _: True, name_prefix=args.name)

    if model_paths:
        os.makedirs(f"results/{args.name}", exist_ok=True)
        with open(f"results/{args.name}/test.csv", "w") as f:
            results = csv.writer(f)
            results.writerow(["id", "length", "success", "spl"])

        for path in tqdm(model_paths):
            if path is not None:
                if args.verbose:
                    print(f"loading {path}")
                model = th.load(path).to(device)
                model.eval()
            else:
                model = None

            infos = play(args.episodes, env, model=model, device=device, hidden=args.hidden, observe=args.observe, delay=args.delay, seed=args.seed, deterministic=args.deterministic)


            s = np.array([info["success"] for info in infos], dtype=float)
            p = np.array([len(info["path"]) for info in infos], dtype=float)
            l = np.array([travel_dist(info["targets"] + [info["initial"]]) + len(info["targets"]) for info in infos], dtype=float)

            spl = np.mean(spl_metric(s, p, l))
            success = np.mean(s)
            length = np.sum(s*p)/np.sum(s)

            with open(f"results/{args.name}/test.csv", "a") as f:
                results = csv.writer(f)
                if path is None:
                    id = "-"
                else:
                    id, _ = os.path.splitext(os.path.basename(path))
                
                results.writerow([id, spl, length, success])

            if args.verbose:
                print(f"{id}: spl: {spl}, length: {length}, success: {success}")
    else:
        infos = play(args.episodes, env, agent=args.agent, hidden=args.hidden, observe=args.observe, delay=args.delay, seed=args.seed)
        
        s = np.array([info["success"] for info in infos], dtype=float)
        p = np.array([len(info["path"]) for info in infos], dtype=float)
        l = np.array([travel_dist(info["targets"] + [info["initial"]]) + len(info["targets"]) for info in infos], dtype=float)

        spl = np.mean(spl_metric(s, p, l))
        success = np.mean(s)
        length = np.sum(s*p)/np.sum(s)
    
        print(f"{args.name}: spl: {spl}, length: {length}, success: {success}")