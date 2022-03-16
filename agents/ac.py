import gym
import numpy as np
import torch as th
import cv2 as cv
import torch.nn as nn
import torch.nn.functional as F
import functools

from torch.distributions import Categorical

from agents.cnn import NatureCNN, AlphaCNN
from agents.utils import preprocess_image, one_hot, init_weights


class Actor(nn.Module):
    def __init__(self, num_features, num_actions, hidden_dims=None):
        super(Actor, self).__init__()

        hidden_dims = [] if hidden_dims is None else hidden_dims
        layer_dims = [num_features] + hidden_dims + [num_actions]
        layers = []

        for i in range(len(layer_dims) - 1):
            in_features = layer_dims[i]
            out_features = layer_dims[i+1]
            
            if i == len(layer_dims) - 2:
                layers.append(init_weights(nn.Linear(in_features, out_features), gain=0.01))
            else:
                layers.append(init_weights(nn.Linear(in_features, out_features)))
                layers.append(nn.Tanh())

        self.policy = nn.Sequential(*layers)

        """
        self.policy = nn.Sequential(
            init_weights(nn.Linear(num_features, 64)),
            nn.Tanh(),
            init_weights(nn.Linear(64, 64)),
            nn.Tanh(),
            init_weights(nn.Linear(64, num_actions), gain=0.01)
        )
        """
    
    def forward(self, x):
        logits = self.policy(x)
        pi = Categorical(logits=logits)
        return pi
    

class Critic(nn.Module):
    def __init__(self, num_features, hidden_dims=None):
        super(Critic, self).__init__()

        hidden_dims = [] if hidden_dims is None else hidden_dims
        layer_dims = [num_features] + hidden_dims + [1]
        layers = []

        for i in range(len(layer_dims) - 1):
            in_features = layer_dims[i]
            out_features = layer_dims[i+1]
            
            if i == len(layer_dims) - 2:
                layers.append(init_weights(nn.Linear(in_features, out_features), gain=1.0))
            else:
                layers.append(init_weights(nn.Linear(in_features, out_features)))
                layers.append(nn.Tanh())

        self.value = nn.Sequential(*layers)

        """
        self.value = nn.Sequential(
            init_weights(nn.Linear(num_features, 64)),
            nn.Tanh(),
            init_weights(nn.Linear(64, 64)),
            nn.Tanh(),
            init_weights(nn.Linear(64, 1), gain=1.0)
        )
        """
    
    def forward(self, x):
        v = self.value(x)
        return v



class Extractor(nn.Module):
    def __init__(self, observation_space, custom_preprocessors=None, custom_extractors=None):
        super(Extractor, self).__init__()
        assert isinstance(observation_space, gym.spaces.Dict)

        preprocessors = {}
        extractors = {}
        features_dim = 0

        for key, space in observation_space.items():
            if isinstance(space, gym.spaces.Box):
                if key == "image":
                    preprocessors[key] = preprocess_image
                    extractors[key] = NatureCNN(space)
                    features_dim += extractors[key].features_dim
                elif key == "overview":
                    preprocessors[key] = preprocess_image
                    extractors[key] = AlphaCNN(space)
                    features_dim += extractors[key].features_dim
                else:
                    extractors[key] = nn.Flatten()
                    features_dim += gym.spaces.flatdim(space)
            elif isinstance(space, gym.spaces.Discrete):
                preprocessors[key] = functools.partial(one_hot, n=space.n)
                features_dim += gym.spaces.flatdim(space)
            else:
                assert False

        self.preprocessors = preprocessors
        self.extractors = nn.ModuleDict(extractors)
        self.features_dim = features_dim


    def forward(self, obs):
        tensors = []

        for key, observation in obs.items():
            x = observation
            
            if key in self.preprocessors:
                x = self.preprocessors[key](x)
            
            if key in self.extractors:
                x = self.extractors[key](x)

            tensors.append(x)
        
        return th.cat(tensors, dim=1)


class ActorCritic(nn.Module):
    def __init__(self, envs):
        super(ActorCritic, self).__init__()

        self.extractor = Extractor(envs.single_observation_space)
        self.network = nn.Identity()

        self.policy = Actor(self.extractor.features_dim, envs.single_action_space.n)#, hidden_dims=[64, 64])
        self.value = Critic(self.extractor.features_dim)#, hidden_dims=[64, 64])

    def forward(self, obs):
        x = self.extractor(obs)
        h = self.network(x)
        pi = self.policy(h)
        v = self.value(h)
        return pi, v

    def predict(self, obs, deterministic=False):
        x = self.extractor(obs)
        h = self.network(x)
        pi = self.policy(h)

        if deterministic:
            return th.argmax(pi.probs).item()
        else:
            return pi.sample().item()


class MemoryActorCritic(nn.Module):
    def __init__(self, envs):
        super(MemoryActorCritic, self).__init__()

        self.extractor = Extractor(envs.single_observation_space)
        self.memory = nn.LSTM(self.extractor.features_dim, 128)

        self.policy = Actor(128, envs.single_action_space.n, hidden_dims=[])
        self.value = Critic(128, hidden_dims=[])

        for name, param in self.memory.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

    def remember(self, x, state, done):
        batch_size = state[0].shape[1]
        x = x.reshape((-1, batch_size, self.memory.input_size))
        done = done.reshape((-1, batch_size))

        new_hidden = []

        for h, d in zip(x, done):
            hidden = (1.0 - d).view(1, -1, 1) * state[0]
            cell = (1.0 - d).view(1, -1, 1) * state[1]
            h, state = self.memory(h.unsqueeze(0), (hidden, cell))
            new_hidden += [h]
        
        new_hidden = th.flatten(th.cat(new_hidden), 0, 1)

        return new_hidden, state

    def forward(self, obs, state, done):        
        x = self.extractor(obs)
        h, s = self.remember(x, state, done)
        pi = self.policy(h)
        v = self.value(h)

        return pi, v, s

    def predict(self, obs, state, done):
        x = self.extractor(obs)
        h, s = self.remember(x, state, done)
        pi = self.policy(h)
        return pi.sample().item(), s
