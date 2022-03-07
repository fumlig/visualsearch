import gym
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical


"""
fuck

we have to be able to store the observations in the ppo buffers
with the current structure extractions have to happen

options:
- rewrite ppo so that it can use an extractor class and run the proper back propagation on it
- then the agent would need a "preprocess/extract method" which we call and get the proper backprop on

- keep flattened observations and use a unified architecture, i.e. use only convolution or mlp for all features
- alphazero did this by having one channel per extra feature, actually not too dumb...
- make an observation transformation wrapper that creates a channel for each key in the dictobs
- discretes are simply filled in
- try and see if it gets the same performance as an mlp first of all
"""


#def init_weights(module, gain=1.0):
#    if isinstance(module, nn.Linear):
#        nn.init.orthogonal_(module.weight, gain)
#        nn.init.constant_(module.bias, 0.0)

def init_weights(layer, gain=np.sqrt(2)):
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer



class Actor(nn.Module):
    def __init__(self, num_features, num_actions):
        super(Actor, self).__init__()

        self.policy = nn.Sequential(
            init_weights(nn.Linear(num_features, 64)),
            nn.Tanh(),
            init_weights(nn.Linear(64, 64)),
            nn.Tanh(),
            init_weights(nn.Linear(64, num_actions), gain=0.01)
        )
    
    def forward(self, x):
        logits = self.policy(x)
        pi = Categorical(logits=logits)
        return pi
    

class Critic(nn.Module):
    def __init__(self, num_features):
        super(Critic, self).__init__()

        self.value = nn.Sequential(
            init_weights(nn.Linear(num_features, 64)),
            nn.Tanh(),
            init_weights(nn.Linear(64, 64)),
            nn.Tanh(),
            init_weights(nn.Linear(64, 1), gain=1.0)
        )
    
    def forward(self, x):
        v = self.value(x)
        return v

"""
class DictExtractor(nn.Module):
    def __init__(self, envs):
        super(ActorCritic, self).__init__()
        obs_space = envs.single_observation_space
        height, width, channels = obs_space["img"].shape
        num_positions = obs_space["pos"].shape

        # convolutions: batch, channels, height, width
        # channels can change between convolution layers

        self.conv_seq = nn.Sequential(
            init_weights(nn.Conv2d(channels, 32, 3, padding=1)),
            nn.ReLU(),
            init_weights(nn.Conv2d(32, 32, 3, padding=1)),
            nn.ReLU(),
            nn.Flatten(),
            init_weights(nn.Linear(height*width*32)),
            nn.ReLU()
        )

    def forward(obs):
        img_hidden = 
"""

class ActorCritic(nn.Module):
    def __init__(self, envs):
        super(ActorCritic, self).__init__()

        num_features, = envs.single_observation_space.shape
        num_actions = envs.single_action_space.n

        print(num_features)

        self.extractor = nn.Identity()
        self.network = nn.Identity()

        self.policy = Actor(num_features, num_actions)
        self.value = Critic(num_features)

        #self.network.apply(lambda m: init_weights(m, np.sqrt(2)))
        #self.actor.apply(lambda m: init_weights(m, 0.01))
        #self.critic.apply(lambda m: init_weights(m, 1.0))

    def forward(self, obs):
        x = self.extractor(obs)
        h = self.network(x)
        pi = self.policy(h)
        v = self.value(h)
        return pi, v

    def predict(self, obs):
        x = self.extractor(obs)
        h = self.network(x)
        pi = self.policy(h)
        return pi.sample().item()
