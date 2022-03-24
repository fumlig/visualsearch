
import gym
import numpy as np
import torch as th
import cv2 as cv
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical

from rl.models import MLP, ImpalaCNN, NatureCNN, AlphaCNN
from rl.utils import preprocess_image, init_lstm



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


# how do we handle triggers for these agents?


class RandomAgent(Agent):
    def __init__(self, env):
        super().__init__()

    def predict(self, _obs, state, **_kwargs):
        return self.action_space.sample(), state


class ExhaustiveAgent(Agent):
    # use some coverage path planning algorithm that finds an optimal covering path given some initial position
    # https://www.sciencedirect.com/science/article/abs/pii/S092188901300167X?via%3Dihub
    def __init__(self, env):
        super().__init__()

    def predict(self, _obs, state, **_kwargs):
        return self.action_space.sample(), state


class SearchAgent(Agent):
    """
    Our method.
    """

    def __init__(self, envs):
        super().__init__(envs)
        assert isinstance(self.observation_space, gym.spaces.Dict)
        assert self.observation_space.get("image") is not None
        assert self.observation_space.get("memory") is not None

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


class ImpalaAgent(Agent):
    """
    Two impala resnets.
    """

    def __init__(self, envs):
        super().__init__(envs)
        assert isinstance(self.observation_space, gym.spaces.Dict)
        assert self.observation_space.get("image") is not None
        assert self.observation_space.get("memory") is not None

        self.image_cnn = ImpalaCNN(self.observation_space["image"])
        self.memory_cnn = ImpalaCNN(self.observation_space["memory"])
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


class ImageAgent(Agent):
    # Mnih et al., 2016 (https://arxiv.org/pdf/1602.01783.pdf)

    # this is basically the one from the atari paper

    def __init__(self, envs):
        super().__init__(envs)
        assert isinstance(self.observation_space, gym.spaces.Dict)
        assert self.observation_space.get("image") is not None

        self.cnn = NatureCNN(self.observation_space["image"])        
        self.policy = MLP(self.cnn.features_dim, self.action_space.n, out_gain=0.01)
        self.value = MLP(self.cnn.features_dim, 1, out_gain=1.0)

    def forward(self, obs, state, **kwargs):
        x = obs["image"]
        x = preprocess_image(x)

        h = self.cnn(x)
        
        logits = self.policy(h)
        pi = Categorical(logits=logits)
        v = self.value(h)

        return pi, v, state



class RecurrentAgent(Agent):    
    # Mnih et al., 2016 (https://arxiv.org/pdf/1602.01783.pdf)

    def __init__(self, envs):
        super().__init__(envs)
        assert isinstance(self.observation_space, gym.spaces.Dict)
        assert self.observation_space.get("image") is not None

        self.cnn = NatureCNN(self.observation_space["image"])
        self.lstm = nn.LSTM(self.cnn.features_dim, 256, num_layers=1)
        
        self.policy = MLP(256, self.action_space.n, out_gain=0.01)
        self.value = MLP(256, 1, out_gain=1.0)

        init_lstm(self.lstm)

    def initial(self, num_envs):
        return (th.zeros(self.lstm.num_layers, num_envs, self.lstm.hidden_size), th.zeros(self.lstm.num_layers, num_envs, self.lstm.hidden_size))

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
        h, state = self.remember(h, state, done)
        
        logits = self.policy(h)
        pi = Categorical(logits=logits)
        v = self.value(h)

        return pi, v, state

    def predict(self, obs, state, done, deterministic=False, **kwargs):
        x = obs["image"]
        x = preprocess_image(x)

        h = self.cnn(x)
        h, state = self.remember(h, state, done)

        logits = self.policy(h)
        pi = Categorical(logits=logits)

        if deterministic:
            return th.argmax(pi.probs).item(), state
        else:
            return pi.sample().item(), state



class BaselineAgent(Agent):    
    # https://arxiv.org/abs/1611.03673


    def __init__(self, envs):
        super().__init__(envs)
        assert isinstance(self.observation_space, gym.spaces.Dict)
        assert self.observation_space.get("image") is not None
        assert self.observation_space.get("last_action") is not None
        #assert self.observation_space.get("last_reward") is not None

        self.cnn = NatureCNN(self.observation_space["image"])
        self.lstm = nn.LSTM(self.cnn.features_dim + self.action_space.n, 256, num_layers=1)
        
        self.policy = MLP(256, self.action_space.n, out_gain=0.01)
        self.value = MLP(256, 1, out_gain=1.0)

        init_lstm(self.lstm)

    def initial(self, num_envs):
        return (th.zeros(self.lstm.num_layers, num_envs, self.lstm.hidden_size), th.zeros(self.lstm.num_layers, num_envs, self.lstm.hidden_size))

    def extract(self, obs):
        xs = []
        xs.append(self.cnn(preprocess_image(obs["image"])))
        xs.append(F.one_hot(obs["last_action"].long(), num_classes=self.action_space.n))
        #xs.append(obs["last_reward"])

        # the authors additionally use relative velocity, but since we have such low-resolution discrete actions the agent should be able to learn this
        # we help it by also encoding the action as a one-hot vector

        return th.cat(xs, dim=1)

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
        x = self.extract(obs)
        h, state = self.remember(x, state, done)
        logits = self.policy(h)
        pi = Categorical(logits=logits)
        v = self.value(h)

        return pi, v, state

    def predict(self, obs, state, done, deterministic=False, **kwargs):
        x = self.extract(obs)
        h, state = self.remember(x, state, done)
        logits = self.policy(h)
        pi = Categorical(logits=logits)

        if deterministic:
            return th.argmax(pi.probs).item(), state
        else:
            return pi.sample().item(), state