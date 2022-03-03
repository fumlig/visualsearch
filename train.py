import random
import json
import os
import numpy as np
import torch as th
import gym
import gym_search

from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser
from agents.ac import ActorCritic
from agents import ppo


# todo: policy loss looks weird as hell

SEED = 0
TOT_TIMESTEPS = int(100e6)
NUM_ENVS = 64 # also a hyperparameter...
HPARAMS = dict(
    learning_rate=5e-4,
    num_steps=256,
    num_minibatches=8,
    num_epochs=3,
    gamma=0.999,
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
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--tot-timesteps", type=int, default=TOT_TIMESTEPS)
    parser.add_argument("--num-envs", type=int, default=NUM_ENVS),
    parser.add_argument("--hparams", type=parse_hparams, default=HPARAMS)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    th.manual_seed(args.seed)

    if args.deterministic:
        th.use_deterministic_algorithms(True)
        #th.backends.cudnn.benchmark = False
        #th.backends.cudnn.deterministic = True

    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    envs = gym.vector.make(args.env_id, args.num_envs, asynchronous=False, wrappers=[gym.wrappers.FlattenObservation, gym.wrappers.RecordEpisodeStatistics])
    #envs = gym.wrappers.NormalizeReward(envs)
    envs.seed(args.seed)
    for env in envs.envs:
        env.action_space.seed(args.seed)
        env.observation_space.seed(args.seed)

    # we can write the feature extractor here!

    agent = ActorCritic(envs)

    writer = SummaryWriter("logs/my")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n" +
        "\n".join([f"|{key}|{value}|" for key, value in args.hparams.items()]) 
    )

    ppo.learn(args.tot_timesteps, envs, agent, device, writer, **args.hparams)

    envs.close()
    writer.close()