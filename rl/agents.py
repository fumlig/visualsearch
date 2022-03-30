
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

    def initial(self, _num_envs):
        return {}

    def forward(self, _obs, _state, **kwargs):
        raise NotImplementedError
    
    def predict(self, _obs, _state, **kwargs):
        raise NotImplementedError


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


# we should have DQN, PPO and A2C
# maybe the actor-critic interface can be used for DQN as well, just ignore value?
# or we use the idea from rllib:
# forward only returns policy,
# optionally stores value as field
# I think this is the way to go


class SearchAgent(Agent):
    """
    Our method.

    # we probably want either fewer outputs from the naturecnn, or a network afterwards whose output is used for the memory only
    # in that case we would have to introduce an auxilliary loss for that head
    # what could this loss be?

    https://arxiv.org/abs/1702.03920
    we will now use a representation that is useful for selecting actions and predicting value
    these authors seem to simply compress the representation with an encoder-decoder, use the compressed representation, and then the decoded one for planning.
    """

    def __init__(self, envs):
        super().__init__(envs)
        assert isinstance(self.observation_space, gym.spaces.Dict)
        assert self.observation_space.get("image") is not None
        assert self.observation_space.get("memory") is not None
        assert self.observation_space.get("position") is not None

        self.image_cnn = NatureCNN(self.observation_space["image"])
        self.latent_dim = self.image_cnn.features_dim

        self.memory_size = self.observation_space["memory"].shape[:2]
        self.memory_channels = self.observation_space["memory"].shape[2] + self.latent_dim
        self.memory_shape = (*self.memory_size, self.memory_channels)
        
        self.memory_cnn = ImpalaCNN(self.memory_shape)
        self.hidden_dim = self.image_cnn.features_dim + self.memory_cnn.features_dim

        self.policy = MLP(self.hidden_dim, self.action_space.n, out_gain=0.01)
        self.value = MLP(self.hidden_dim, 1, out_gain=1.0)

    def initial(self, num_envs):
        return [th.zeros((num_envs, *self.memory_size, self.latent_dim))]

    def extract(self, obs, state):
        image = obs["image"]
        latent_image = self.image_cnn(preprocess_image(image))

        position = obs["position"].long()
        state_memory, = state
        state_memory[:,position[:,0],position[:,1]] = latent_image #.clone().detach()
        
        obs_memory = obs["memory"]
        batch_size = obs_memory.shape[0]

        # obs memory is much larger here

        print(state_memory.shape, obs_memory.shape)

        memory = th.cat([state_memory.expand(batch_size, -1, -1, -1), obs_memory], axis=-1)
        latent_memory = self.memory_cnn(preprocess_image(memory))

        return th.cat([latent_image, latent_memory], dim=1), {"memory": state_memory}

    def forward(self, obs, state, **kwargs):
        x, state = self.extract(obs, state)
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
        return [th.zeros(self.lstm.num_layers, num_envs, self.lstm.hidden_size), th.zeros(self.lstm.num_layers, num_envs, self.lstm.hidden_size)]

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
        assert self.observation_space.get("position") is not None
        #assert self.observation_space.get("last_action") is not None
        #assert self.observation_space.get("last_reward") is not None

        self.cnn = NatureCNN(self.observation_space["image"])
        
        hidden_dim = 0
        hidden_dim += self.cnn.features_dim
        #hidden_dim += self.observation_space["position"][0].n
        #hidden_dim += self.observation_space["position"][1].n
        hidden_dim += self.action_space.n
        #hidden_dim += 1

        self.lstm = nn.LSTM(hidden_dim, 256, num_layers=1)

        self.policy = MLP(256, self.action_space.n, out_gain=0.01)
        self.value = MLP(256, 1, out_gain=1.0)

        init_lstm(self.lstm)

    def initial(self, num_envs):
        return [th.zeros(self.lstm.num_layers, num_envs, self.lstm.hidden_size), th.zeros(self.lstm.num_layers, num_envs, self.lstm.hidden_size)]

    def extract(self, obs):
        xs = []

        xs.append(self.cnn(preprocess_image(obs["image"])))
        #xs.append(F.one_hot(obs["position"][:,0].long(), num_classes=self.observation_space["position"][0].n))
        #xs.append(F.one_hot(obs["position"][:,1].long(), num_classes=self.observation_space["position"][1].n))
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