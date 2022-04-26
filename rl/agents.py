
import gym
import numpy as np
import torch as th
import cv2 as cv
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical

from rl.models import MLP, NatureCNN, ImpalaCNN, NeuralMap, SimpleMap
from rl.utils import preprocess_image, init_lstm



class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.observation_space = envs.single_observation_space if envs.is_vector_env else envs.observation_space
        self.action_space = envs.single_action_space if envs.is_vector_env else envs.action_space

    def initial(self, _num_envs):
        return []

    def forward(self, _obs, _state, **kwargs):
        raise NotImplementedError
    
    def predict(self, obs, state, deterministic=False, **kwargs):
        pi, _, state = self.forward(obs, state, **kwargs)
        
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
        self.policy = MLP(self.cnn.output_dim, self.action_space.n, out_gain=0.01)
        self.value = MLP(self.cnn.output_dim, 1, out_gain=1.0)

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
        self.lstm = nn.LSTM(self.cnn.output_dim, 256, num_layers=1)
        
        self.policy = MLP(256, self.action_space.n, out_gain=0.01)
        self.value = MLP(256, 1, out_gain=1.0)

        init_lstm(self.lstm)

    def initial(self, num_envs):
        return [th.zeros(num_envs, self.lstm.num_layers, self.lstm.hidden_size), th.zeros(num_envs, self.lstm.num_layers, self.lstm.hidden_size)]

    def remember(self, hidden, state, done):
        state = [s.transpose(0, 1) for s in state]
        batch_size = state[0].shape[1]
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []

        for h, d in zip(hidden, done):
            h, state = self.lstm(h.unsqueeze(0), ((1.0 - d).view(1, -1, 1) * state[0], (1.0 - d).view(1, -1, 1) * state[1]))
            new_hidden += [h]

        new_hidden = th.flatten(th.cat(new_hidden), 0, 1)
        state = [s.transpose(0, 1) for s in state]

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


class LSTMAgent(Agent):    
    # https://arxiv.org/abs/1611.03673


    def __init__(self, envs, num_layers=1, dropout=0.0):
        super().__init__(envs)
        assert isinstance(self.observation_space, gym.spaces.Dict)
        assert self.observation_space.get("image") is not None
        assert self.observation_space.get("position") is not None

        self.cnn = NatureCNN(self.observation_space["image"])
        
        hidden_dim = self.cnn.output_dim + self.observation_space["position"][0].n + self.observation_space["position"][1].n
        #hidden_dim = self.cnn.output_dim + self.observation_space["position"][0].n*self.observation_space["position"][1].n

        self.lstm = nn.LSTM(hidden_dim, 128, num_layers=num_layers, dropout=dropout)

        self.policy = MLP(128, self.action_space.n, out_gain=0.01)
        self.value = MLP(128, 1, out_gain=1.0)

        init_lstm(self.lstm)

    def initial(self, num_envs):
        return [th.zeros(num_envs, self.lstm.num_layers, self.lstm.hidden_size), th.zeros(num_envs, self.lstm.num_layers, self.lstm.hidden_size)]

    def extract(self, obs):
        x = preprocess_image(obs["image"])
        h = self.cnn(x)
        
        p = th.cat([F.one_hot(obs["position"][:,0].long(), num_classes=self.observation_space["position"][0].n), F.one_hot(obs["position"][:,1].long(), num_classes=self.observation_space["position"][1].n)], dim=1)
        #p = F.one_hot(obs["position"][:,0].long()*self.observation_space["position"][1].n + obs["position"][:,1].long(), num_classes=self.observation_space["position"][0].n*self.observation_space["position"][1].n)

        return th.cat([h, p], dim=1)

    def remember(self, hidden, state, done):
        state = [s.transpose(0, 1).contiguous() for s in state]
        batch_size = state[0].shape[1]
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []

        assert(len(hidden) == len(done))

        for h, d in zip(hidden, done):
            h, state = self.lstm(h.unsqueeze(0), ((1.0 - d).view(1, -1, 1) * state[0], (1.0 - d).view(1, -1, 1) * state[1]))
            new_hidden += [h]
        
        new_hidden = th.flatten(th.cat(new_hidden), 0, 1)
        state = [s.transpose(0, 1) for s in state]
        return new_hidden, state

    def forward(self, obs, state, done, **kwargs):
        x = self.extract(obs)
        h, state = self.remember(x, state, done)
        logits = self.policy(h)
        pi = Categorical(logits=logits)
        v = self.value(h)

        return pi, v, state


class MapAgent(Agent):

    def __init__(self, envs):
        super().__init__(envs)
        assert isinstance(self.observation_space, gym.spaces.Dict)
        assert self.observation_space.get("image") is not None
        assert self.observation_space.get("position") is not None

        self.image_cnn = NatureCNN(self.observation_space["image"])        
        self.map_net = SimpleMap([s.n for s in self.observation_space["position"]], self.image_cnn.output_dim)

        self.policy = MLP(self.map_net.output_dim, self.action_space.n, out_gain=0.01)
        self.value = MLP(self.map_net.output_dim, 1, out_gain=1.0)

    def initial(self, num_envs):
        return [th.zeros((num_envs, *self.map_net.shape))]

    def forward(self, obs, state, done, **kwargs):
        hidden = self.image_cnn(preprocess_image(obs["image"]))
        index = obs["position"].long()
    
        state = state[0]
        batch_size = state.shape[0]
        hidden = hidden.reshape((-1, batch_size, self.map_net.input_dim))
        done = done.reshape((-1, batch_size))
        index = index.reshape((-1, batch_size, 2))
        new_hidden = []

        for h, d, i in zip(hidden, done, index):
            masked = (1.0 - d).view(-1, 1, 1, 1)*state
            h, state = self.map_net(h, masked, i)
            new_hidden.append(h)

        state = [state]
        hidden = th.cat(new_hidden, dim=0)      

        logits = self.policy(hidden)
        pi = Categorical(logits=logits)
        v = self.value(hidden)

        return pi, v, state


AGENTS = {
    "lstm": LSTMAgent,
    "map": MapAgent
}


def make(id, envs, **kwargs):
    cls = AGENTS.get(id)
    
    if cls is None:
        return None
    
    return cls(envs, **kwargs)