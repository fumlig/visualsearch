#!/usr/bin/env python3

from collections import deque
import random
import yaml
import os
import datetime as dt
import numpy as np
import torch as th
import pandas as pd
import gym

from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser
from tqdm import tqdm

import gym_search
import rl


SEED = 0
TOT_TIMESTEPS = int(25e6)
NUM_ENVS = 4


def parse_hparams(s):
    if os.path.exists(s):
        with open(s) as f:
            return yaml.load(f, Loader=yaml.FullLoader)

    return yaml.safe_load(s)

def env_default(key, default=None):
    value = os.environ.get(key)

    if value is None:
        value = default

    if value is None:
        return dict()
    
    return dict(default=value, nargs='?')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("environment", type=str, **env_default("ENV_ID"))
    parser.add_argument("agent", type=str, choices=rl.agents.AGENTS.keys())
    parser.add_argument("algorithm", type=str, choices=rl.algorithms.ALGORITHMS.keys())

    parser.add_argument("--num-timesteps", type=int, default=TOT_TIMESTEPS)
    parser.add_argument("--num-envs", type=int, default=NUM_ENVS)
    parser.add_argument("--env-kwargs", type=parse_hparams, default={})
    parser.add_argument("--agent-kwargs", type=parse_hparams, default={})
    parser.add_argument("--alg-kwargs", type=parse_hparams, default={})

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--name", type=str)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--model", type=str)
    parser.add_argument("--eval-interval", type=int, default=10000)
    parser.add_argument("--ckpt-interval", type=int, default=100000)


    args = parser.parse_args()

    if args.name is None:
        args.name = f"{args.environment.lower()}-{args.algorithm}-{args.agent}-{args.seed}-{dt.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        th.manual_seed(args.seed)

    if args.deterministic:
        th.use_deterministic_algorithms(True)
        #th.backends.cudnn.benchmark = False
        #th.backends.cudnn.deterministic = True

    wrappers = [
        #gym_search.wrappers.LastAction,
        #gym_search.wrappers.LastReward,
    ]

    envs = gym.vector.make(args.environment, args.num_envs, asynchronous=False, wrappers=wrappers, **args.env_kwargs)
    envs = gym.wrappers.RecordEpisodeStatistics(envs)
    envs = gym.wrappers.NormalizeReward(envs)

    device = th.device(args.device)
    agent = th.load(args.model) if args.model else rl.agents.make(args.agent, envs, **args.agent_kwargs)

    df = pd.DataFrame()

    writer = SummaryWriter(f"logs/{args.name}/train")
    writer.add_text(
        "agent/hyperparameters",
        "|parameter|value|\n" +
        "|---|---|\n" +
        f"|num_timesteps|{args.num_timesteps}|\n" +
        f"|num_envs|{args.num_envs}|\n" +
        "".join([f"{param}|{value}\n" for param, value in args.env_kwargs.items()]) +
        "".join([f"{param}|{value}\n" for param, value in args.alg_kwargs.items()]) +
        "".join([f"{param}|{value}\n" for param, value in args.agent_kwargs.items()]) +
        f"|seed|{args.seed}|\n"
    )

    pbar = tqdm(total=args.num_timesteps)
    ep_queue = deque(maxlen=100)
    last_timestep = 0

    for timestep, info in rl.algorithms.learn(args.algorithm, args.num_timesteps, envs, agent, device, seed=args.seed, **args.alg_kwargs):

        if "batch" in info:
            pbar.update(timestep - pbar.n)

        if "episode" in info:
            writer.add_scalar("episode/return", info["episode"]["r"], timestep)
            writer.add_scalar("episode/length", info["episode"]["l"], timestep)

            ep_queue.append({
                "return": info["episode"]["r"],
                "length": info["episode"]["l"],
                "success": info["success"],
            })

        if "loss" in info:
            for key, value in info["loss"].items():
                writer.add_scalar(f"loss/{key}", value, timestep)

        if "counter" in info:
            for key, value in info["counter"].items():
                writer.add_scalar(f"counter/{key}", value, timestep)

        if args.eval_interval and timestep // args.eval_interval > last_timestep // args.eval_interval:
            eval_step = (timestep // args.eval_interval) * args.eval_interval

            avg_return = np.mean([ep["return"] for ep in ep_queue])
            avg_length = np.mean([ep["length"] for ep in ep_queue])
            avg_success = np.mean([ep["success"] for ep in ep_queue])

            writer.add_scalar("average/return", avg_return, eval_step)
            writer.add_scalar("average/length", avg_length, eval_step)
            writer.add_scalar("average/success", avg_success, eval_step)

            pbar.set_description(f"ret {avg_return:.2f}, len {avg_length:.2f}")

        if args.ckpt_interval and timestep // args.ckpt_interval > last_timestep // args.ckpt_interval:
            ckpt_step = (timestep // args.ckpt_interval) * args.ckpt_interval

            os.makedirs(os.path.dirname(f"models/{args.name}/ckpt/{timestep}.pt"), exist_ok=True)
            th.save(agent, f"models/{args.name}-ckpt-{timestep}.pt")

        last_timestep = timestep

    pbar.update(args.num_timesteps)

    print(f"saving model models/{args.name}.pt")
    os.makedirs(os.path.dirname(f"models/{args.name}.pt"), exist_ok=True)
    th.save(agent, f"models/{args.name}.pt")

    envs.close()
    writer.close()
