
import gym
import numpy as np
import torch as th
import cv2 as cv
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical

from agents.mlp import MLP
from agents.cnn import NatureCNN, AlphaCNN
from agents.utils import preprocess_image, one_hot, init_weights, init_lstm



class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.observation_space = envs.single_observation_space
        self.action_space = envs.single_action_space

    def initial(self, num_envs):
        return (th.empty(0, num_envs), th.empty(0, num_envs))

    def forward(self, _obs, _state, **kwargs):
        raise NotImplementedError
    
    def predict(self, _obs, _state, **kwargs):
        raise NotImplementedError


class SearchAgent(Agent):
    """
    Our method.
    """

    def __init__(self, envs):
        super().__init__(envs)
        assert isinstance(self.observation_space, gym.spaces.Dict)
        assert self.observation_space.get("image") is not None
        assert self.observation_space.get("memory") is not None

        # todo: we could use ImpalaCNN for both instead
        # check training time though

        self.image_cnn = NatureCNN(self.observation_space["image"])
        self.memory_cnn = AlphaCNN(self.observation_space["memory"])
        self.features_dim = self.image_cnn.features_dim + self.memory_cnn.features_dim

        self.policy = MLP(self.features_dim, self.action_space.n, out_gain=0.01)
        self.value = MLP(self.features_dim, 1, out_gain=1.0)

    def extract(self, obs):
        xs = []
        xs.append(self.image_cnn(preprocess_image(obs["image"])))
        xs.append(self.memory_cnn(preprocess_image(obs["memory"])))
        return th.cat(xs, dim=1)

    def forward(self, obs, state, **kwargs):
        x = self.extract(obs)
        logits = self.policy(x)
        pi = Categorical(logits=logits)
        v = self.value(x)
        return pi, v, state

    def predict(self, obs, state, deterministic=False, **kwargs):
        x = self.extract(obs)
        logits = self.policy(x)
        pi = Categorical(logits=logits)

        if deterministic:
            return th.argmax(pi.probs).item(), state
        else:
            return pi.sample().item(), state


class MemoryAgent(Agent):
    """
    Baseline method.
    
    Mnih et al., 2016 (https://arxiv.org/pdf/1602.01783.pdf)
    """

    def __init__(self, envs):
        super().__init__(envs)
        assert isinstance(self.observation_space, gym.spaces.Dict)
        assert "image" in self.observation_space

        self.cnn = NatureCNN(self.observation_space)
        self.lstm = nn.LSTM(self.cnn.features_dim, 256)
        
        self.policy = MLP(256, self.action_space.n, out_gain=0.01)
        self.value = MLP(256, 1, out_gain=1.0)

        init_lstm(self.lstm)

    def initial(self, num_envs):
        return th.zeros(self.lstm.num_layers, num_envs, self.lstm.hidden_size)

    def remember(self, hidden, state, done):
        batch_size = state[0].shape[1]
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []

        for h, d in zip(hidden, done):
            hidden = (1.0 - d).view(1, -1, 1) * state[0]
            cell = (1.0 - d).view(1, -1, 1) * state[1]
            h, state = self.lstm(h.unsqueeze(0), (hidden, cell))
            new_hidden += [h]
        
        new_hidden = th.flatten(th.cat(new_hidden), 0, 1)

        return new_hidden, state

    def forward(self, obs, state, done, **kwargs):
        x = obs["image"]
        x = preprocess_image(x)
        
        h = self.cnn(x)
        h, s = self.remember(h, state, done)
        
        logits = self.policy(h)
        pi = Categorical(logits=logits)
        v = self.value(h)

        return pi, v, s


class RandomAgent():
    def __init__(self, env):
        self.action_space = env.action_space

    def predict(self, _obs):
        return self.action_space.sample()

