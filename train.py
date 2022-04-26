#!/usr/bin/env python3

import random
import yaml
import os
import datetime as dt
import signal
import numpy as np
import torch as th
import gym

from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser
from skopt import gp_minimize

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


def sigint_handler(signum, frame, close, **kwargs):
    close(**kwargs)
    exit(0)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("environment", type=str, **env_default("ENV_ID"))
    parser.add_argument("algorithm", type=str, choices=rl.algorithms.ALGORITHMS.keys())
    parser.add_argument("agent", type=str, choices=rl.agents.AGENTS.keys())

    parser.add_argument("--num-timesteps", type=int, default=TOT_TIMESTEPS)
    parser.add_argument("--num-envs", type=int, default=NUM_ENVS)
    parser.add_argument("--env-kwargs", type=parse_hparams, default={})
    parser.add_argument("--alg-kwargs", type=parse_hparams, default={})
    parser.add_argument("--agent-kwargs", type=parse_hparams, default={})

    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--name", type=str)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--model", type=str)
    parser.add_argument("--ckpt-interval", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    #parser.add_argument("--tune", type=parse_hparams)

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

    #if args.tune:
    #    gp_minimize(...)

    device = th.device("cuda" if th.cuda.is_available() and not args.cpu else "cpu")
    writer = SummaryWriter(f"logs/train/{args.name}")

    # environment
    wrappers = [
        #gym_search.wrappers.LastAction,
        #gym_search.wrappers.LastReward,
    ]

    envs = gym.vector.make(args.environment, args.num_envs, asynchronous=False, wrappers=wrappers, **args.env_kwargs)
    envs = gym.wrappers.RecordEpisodeStatistics(envs)
    envs = gym.wrappers.NormalizeReward(envs)

    # agent
    agent = th.load(args.model) if args.model else rl.agents.make(args.agent, envs, **args.agent_kwargs)

    # train
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

    def close():
        envs.close()
        writer.close()
        print(f"saving model models/{args.name}.pt")
        th.save(agent, f"models/{args.name}.pt")

    last_timestep = 0

    def callback(agent, timestep):
        global last_timestep

        if not args.ckpt_interval:
            return

        this_ckpt = timestep // args.ckpt_interval
        last_ckpt = last_timestep // args.ckpt_interval
        
        if this_ckpt > last_ckpt:
            print(f"saving checkpoint models/{args.name}-ckpt-{timestep}.pt")
            th.save(agent, f"models/{args.name}-ckpt-{timestep}.pt")

        last_timestep = timestep

    signal.signal(signal.SIGINT, lambda signum, frame: sigint_handler(signum, frame, close=close))

    rl.algorithms.learn(args.algorithm, args.num_timesteps, envs, agent, device, writer, seed=args.seed, callback=callback, **args.alg_kwargs)

    close()

