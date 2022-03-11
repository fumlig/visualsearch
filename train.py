#!/usr/bin/env python3

import random
import json
import os
import datetime as dt
import numpy as np
import torch as th
import gym
import gym_search

from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser
from gym_search.wrappers import ResizeImage
from agents import ac
from agents import ppo


import gym.wrappers

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
TOT_TIMESTEPS = int(10e6)
NUM_ENVS = 8 # also a hyperparameter...
ALG_PARAMS = dict(
    learning_rate=2.5e-4,
    num_steps=256,
    num_minibatches=4,
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
    parser.add_argument("env_id", type=str, **env_default("ENV_ID"))
    parser.add_argument("--name", type=str, default=dt.datetime.now().isoformat())
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--load", type=str)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--tot-timesteps", type=int, default=TOT_TIMESTEPS)
    parser.add_argument("--num-envs", type=int, default=NUM_ENVS),
    parser.add_argument("--alg-params", type=parse_hparams, default=ALG_PARAMS)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    th.manual_seed(args.seed)

    if args.deterministic:
        th.use_deterministic_algorithms(True)
        #th.backends.cudnn.benchmark = False
        #th.backends.cudnn.deterministic = True

    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    wrappers = [gym.wrappers.RecordEpisodeStatistics, ResizeImage]#, ObservePosition, ObserveTime, ObserveVisible, ObserveVisited]
    envs = gym.vector.make(args.env_id, args.num_envs, asynchronous=False, wrappers=wrappers)
    #envs = gym.wrappers.NormalizeReward(envs)
    envs.seed(args.seed)
    for env in envs.envs:
        env.action_space.seed(args.seed)
        env.observation_space.seed(args.seed)

    if args.load:
        agent = th.load(args.load)
    else:
        agent = ac.ActorCritic(envs)

    writer = SummaryWriter(f"logs/{args.name}")

    ppo.learn(args.tot_timesteps, envs, agent, device, writer, **args.alg_params)

    envs.close()
    writer.close()

    print(f"saving as {args.name}")
    th.save(agent, f"models/{args.name}.pt")