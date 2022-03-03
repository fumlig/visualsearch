import random
import numpy as np
import torch as th
import gym
import gym_search

from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser

from agents.ac import ActorCritic
from agents.ppo import learn


# todo: policy loss looks weird as hell


SEED =  0
ENV_ID = "SearchDense-v0"
NUM_ENVS = 64 # also a hyperparameter...
HPARAMS = dict(
    learning_rate=5e-4,
    tot_timesteps=int(100e6),
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


if __name__ == "__main__":
    parser = ArgumentParser()

    random.seed(SEED)
    np.random.seed(SEED)
    th.manual_seed(SEED)
    th.backends.cudnn.deterministic = True

    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    envs = gym.vector.make(
        ENV_ID,
        NUM_ENVS,
        asynchronous=False,
        wrappers=[gym.wrappers.FlattenObservation, gym.wrappers.RecordEpisodeStatistics]
    )
    envs = gym.wrappers.NormalizeReward(envs)

    envs.seed(SEED)

    for env in envs.envs:
        env.action_space.seed(SEED)
        env.observation_space.seed(SEED)

    agent = ActorCritic(envs).to(device)

    writer = SummaryWriter("logs/my")

    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n" +
        "\n".join([f"|{param}|{value}|" for param, value in HPARAMS.items()]) 
    )

    learn(envs, agent, device, writer, **HPARAMS)

    envs.close()
    writer.close()