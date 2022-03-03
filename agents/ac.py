import gym
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical


#def init_weights(module, gain=1.0):
#    if isinstance(module, nn.Linear):
#        nn.init.orthogonal_(module.weight, gain)
#        nn.init.constant_(module.bias, 0.0)

def init_weights(layer, gain=np.sqrt(2)):
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


def is_image_space(obs_space):
    return isinstance(obs_space, gym.spaces.Box) and len(obs_space.shape == 3) and obs_space.shape[2] == 3 and obs_space.dtype == np.uint8


# https://github.com/AIcrowd/neurips2020-procgen-starter-kit/blob/142d09586d2272a17f44481a115c4bd817cf6a94/models/impala_cnn_torch.py
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)

    def forward(self, x):
        inputs = x
        x = F.relu(x)
        x = self.conv0(x)
        x = F.relu(x)
        x = self.conv1(x)
        return x + inputs


class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=self._input_shape[0], out_channels=self._out_channels, kernel_size=3, padding=1)
        self.res0 = ResidualBlock(self._out_channels)
        self.res1 = ResidualBlock(self._out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res0(x)
        x = self.res1(x)
        assert x.shape[1:] == self.output_shape()
        return x

    def output_shape(self):
        _c, h, w = self._input_shape
        return (self._out_channels, (h + 1) // 2, (w + 1) // 2)

"""
class FeatureExtractor(nn.Module):
    import gym.spaces
    def __init__(self, obs_space):
        if isinstance(obs_space, gym.spaces.Box) and is_image_space(obs_space):
            self.preprocess = None
            self.extract = None # conv
        elif isinstance(obs_space, gym.spaces.Dict):
            self.extract = nn.ModuleDict({name: FeatureExtractor(space) for name, space in obs_space})
        else:
            self.preprocess = gym.spaces.flatten
            self.extract = None

    def forward(self, obs):
        img = obs["img"] # conv
        pos = obs["pos"] # encode as one-hot
"""



class ActorCritic(nn.Module):
    def __init__(self, envs):
        super(ActorCritic, self).__init__()

        num_features, = envs.single_observation_space.shape
        num_actions = envs.single_action_space.n

        #self.network = nn.Identity()
        self.network = nn.Sequential()

        self.actor = nn.Sequential(
            init_weights(nn.Linear(num_features, 64)),
            nn.Tanh(),
            init_weights(nn.Linear(64, 64)),
            nn.Tanh(),
            init_weights(nn.Linear(64, num_actions), gain=0.01)
        )

        self.critic = nn.Sequential(
            init_weights(nn.Linear(num_features, 64)),
            nn.Tanh(),
            init_weights(nn.Linear(64, 64)),
            nn.Tanh(),
            init_weights(nn.Linear(64, 1), gain=1.0)
        )

        #self.network.apply(lambda m: init_weights(m, np.sqrt(2)))
        #self.actor.apply(lambda m: init_weights(m, 0.01))
        #self.critic.apply(lambda m: init_weights(m, 1.0))

    def forward(self, obs):
        hid = self.network(obs)
        pi = self.policy(hid)
        v = self.value(hid)
        return pi, v

    def predict(self, obs):
        hid = self.network(obs)
        pi = self.policy(hid)
        return pi.sample()

    def policy(self, hid):
        logits = self.actor(hid)
        pi = Categorical(logits=logits)
        return pi

    def value(self, hid):
        v = self.critic(hid)
        return v
