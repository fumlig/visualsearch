#!/usr/bin/env python3

import argparse
import datetime
from tkinter import E

import gym
import gym_search

import torch as th
import torch.nn as nn

from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CombinedExtractor(BaseFeaturesExtractor):
    """
    Combined feature extractor for Dict observation spaces.
    Builds a feature extractor for each key of the space. Input from each space
    is fed through a separate submodule (CNN or MLP, depending on input shape),
    the output features are concatenated and fed through additional MLP network ("combined").

    :param observation_space:
    :param cnn_output_dim: Number of features to output from each CNN submodule(s). Defaults to
        256 to avoid exploding network sizes.
    """

    def __init__(self, observation_space: gym.spaces.Dict, cnn_output_dim: int = 256):
        # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
        super(CombinedExtractor, self).__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
                # The observation key is a vector, flatten it if needed
                extractors[key] = nn.Flatten()
                total_concat_size += gym.spaces.utils.flatdim(subspace)

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return th.cat(encoded_tensor_list, dim=1)


parser = argparse.ArgumentParser()
parser.add_argument("env", type=str)
parser.add_argument("-p", "--policy", type=str, choices=["MlpPolicy", "CnnPolicy", "MultiInputPolicy"], default="MultiInputPolicy")
parser.add_argument("-l", "--logs", type=str, default="logs")

args = parser.parse_args()
env = make_vec_env(args.env, n_envs=8)#, wrapper_class=gym.wrappers.FlattenObservation)
name = f"ppo-{args.policy.lower()}-{datetime.datetime.now().isoformat()}"

checkpoint_callback = CheckpointCallback(save_freq=100000, save_path=args.logs, name_prefix='ckpt')

policy_kwargs = dict(
    features_extractor_class=CombinedExtractor,
    features_extractor_kwargs=dict()
)

model = PPO(args.policy, env, verbose=1, tensorboard_log=args.logs, ent_coef=0.01, policy_kwargs=policy_kwargs)
model.learn(total_timesteps=int(50e6), tb_log_name=name, callback=checkpoint_callback)
model.save(name)
"""
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
"""