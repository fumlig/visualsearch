import gym
import numpy as np
import torch as th
import torch.nn as nn

from torch.distributions import Categorical


def init_weights(module, gain=1.0):
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain)
        nn.init.constant_(module.bias, 0.0)


class ActorCritic(nn.Module):
    def __init__(self, envs):
        super(ActorCritic, self).__init__()

        num_features, = envs.single_observation_space.shape
        num_actions = envs.single_action_space.n

        self.network = nn.Identity()
    
        self.actor = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, num_actions)
        )

        self.critic = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.network.apply(lambda m: init_weights(m, np.sqrt(2)))
        self.actor.apply(lambda m: init_weights(m, 1.0))
        self.critic.apply(lambda m: init_weights(m, 0.01))

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