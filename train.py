#!/usr/bin/env python3

import random
import json
import os
import datetime as dt
import numpy as np
import torch as th
import gym

from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser

import gym_search
import rl


"""
todo:
fix hparams (defaults and all that, if we set to {} they are not visible in tensorboard)
fix dict observations
add checkpoints
add video recording
add pretty plotting (yield info from learn?)
"""


# these are from procgen!

SEED = 0
TOT_TIMESTEPS = int(25e6)
NUM_ENVS = 64 # 64 in procgen
HPARAMS = dict(
    learning_rate=5e-4,
    num_steps=256, # 256 in procgen, recommended to be much smaller than episode length 
    num_minibatches=8,
    num_epochs=4,
    gamma=0.99,
    gae_lambda=0.95,
    norm_adv=True,
    clip_range=0.2,
    clip_vloss=True,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    target_kl=None
)


def parse_hparams(s):
    if os.path.exists(s):
        with open(s, "r") as f:
            return json.load(f)
    
    return json.loads(s)

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
    parser.add_argument("algorithm", type=str, choices=rl.ALGORITHMS.keys())
    parser.add_argument("agent", type=str, choices=rl.AGENTS.keys())

    parser.add_argument("--name", type=str)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--model", type=str)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--tot-timesteps", type=int, default=TOT_TIMESTEPS)
    parser.add_argument("--num-envs", type=int, default=NUM_ENVS),
    parser.add_argument("--hparams", type=parse_hparams, default=HPARAMS)

    args = parser.parse_args()

    if args.name is None:
        args.name = f"{args.environment.lower()}-{args.algorithm}-{args.agent}-{dt.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        th.manual_seed(args.seed)

    if args.deterministic:
        th.use_deterministic_algorithms(True)
        #th.backends.cudnn.benchmark = False
        #th.backends.cudnn.deterministic = True

    wrappers = [
        gym.wrappers.RecordEpisodeStatistics,
        gym_search.wrappers.ResizeImage,
        #gym_search.wrappers.ExplicitMemory,
        gym_search.wrappers.LastAction,
        #gym_search.wrappers.LastReward,
    ]

    envs = gym.vector.make(args.environment, args.num_envs, asynchronous=False, wrappers=wrappers)
    envs = gym.wrappers.NormalizeReward(envs)
    
    envs.seed(args.seed)
    for env in envs.envs:
        env.action_space.seed(args.seed)
        env.observation_space.seed(args.seed)

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    writer = SummaryWriter(f"logs/{args.name}")
    agent = rl.agent(args.agent)(envs)
    algorithm = rl.algorithm(args.algorithm)(**args.hparams)

    if args.model:
        agent = th.load(args.model)

    stats = algorithm.learn(args.tot_timesteps, envs, agent, device, writer)

    envs.close()
    writer.close()

    print(f"saving as {args.name}")
    th.save(agent, f"models/{args.name}.pt")
    stats.wr