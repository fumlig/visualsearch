#!/usr/bin/env python3

"""
Test a search agent.
"""

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

import rl_library as rl
import gym_search

from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter
from time import process_time
from tqdm import tqdm
from gym_search.utils import travel_dist


KEY_ESC = 27
KEY_RET = 13
WINDOW_SIZE = (640, 640)

BASELINES = ["human", "greedy", "random", "exhaustive", "handcrafted"]
ENVIRONMENTS = {"gaussian": "Gaussian-v0", "terrain": "Terrain-v0", "camera": "Camera-v0"}


def metrics(infos):
    success = np.array([info["success"] for info in infos], dtype=float)
    taken = np.array([len(info["path"]) for info in infos], dtype=float)
    shortest = np.array([travel_dist(info["targets"] + [info["initial"]]) + len(info["targets"]) for info in infos], dtype=float)

    return {
        # https://arxiv.org/pdf/1807.06757.pdf
        "spl": np.mean(success*shortest/np.maximum(shortest, taken)), 
        "length": np.sum(success*taken)/np.sum(success),
        "success": np.mean(success),
        "triggers": np.mean([info["counter"]["triggers"] for info in infos]),
        "explored": np.mean([info["counter"]["explored"] for info in infos]),
        "revisits": np.mean([info["counter"]["revisits"] for info in infos]),
    }


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
                    obs = {key: th.tensor(sub_obs.copy()).float().unsqueeze(0).to(device) for key, sub_obs in obs.items()}
                    act, state = model.predict(obs, state, done=th.tensor(done).float().unsqueeze(0).to(device), deterministic=deterministic)
                    act = act.item()
            elif agent == "human":
                act = env.get_keys_to_action().get((key,), 0)
            elif agent == "random":
                act = env.get_random_action()
            elif agent == "greedy":
                act = env.get_greedy_action()
            elif agent == "exhaustive":
                act = env.get_exhaustive_action()
            elif agent == "handcrafted":
                act = env.get_handcrafted_action()
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
    parser.add_argument("environment", type=str, choices=ENVIRONMENTS.keys())
    parser.add_argument("--env-kwargs", type=parse_hparams, default={})

    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--runs", type=int, default=1)

    parser.add_argument("--agent", type=str, choices=BASELINES, default="human")
    parser.add_argument("--models", type=str, nargs="*", default=[])
    
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--name", type=str, default=dt.datetime.now().isoformat())
    parser.add_argument("--deterministic", action="store_true")
    
    parser.add_argument("--delay", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--observe", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--hidden", action="store_true")

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

    env = gym.make(ENVIRONMENTS[args.environment], **args.env_kwargs)
    for wrapper in wrappers:
        env = wrapper(env)

    device = th.device(args.device)
    df = pd.DataFrame()

    if args.name is None:
        args.name = dt.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')

    if args.record:
        env = gym.wrappers.RecordVideo(env, "videos", episode_trigger=lambda _: True, name_prefix=args.name)

    if args.deterministic and args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        th.manual_seed(args.seed)

    run_metrics = []

    if model_paths:
        for path in tqdm(model_paths):
            model = th.load(path).to(device)
            model.eval()

            id, _ = os.path.splitext(os.path.basename(path))
            infos = play(args.episodes, env, model=model, device=device, hidden=args.hidden, observe=args.observe, delay=args.delay, seed=args.seed)
            scores = metrics(infos)
            scores.update({"id": id})
            run_metrics.append(scores)

            if args.verbose:
                print(f"{args.name}/{id}:", metrics(infos))

    else:
        for run in tqdm(range(args.runs)):
            infos = play(args.episodes, env, agent=args.agent, hidden=args.hidden, observe=args.observe, delay=args.delay, seed=args.seed)
            scores = metrics(infos)
            scores.update({"id": run})
            run_metrics.append(scores)

            if args.verbose:
                print(f"{args.name}/{run}:", metrics(infos))


    os.makedirs(f"results/{args.name}", exist_ok=True)
    
    with open(f"results/{args.name}/test.csv", "w") as f:
        results = csv.DictWriter(f, fieldnames=run_metrics[0].keys())
        results.writeheader()

        for row in run_metrics:
            results.writerow(row)

            if args.verbose:
                print(row)