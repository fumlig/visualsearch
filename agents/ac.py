import gym
import numpy as np
import torch as th
import torch.nn as nn

from torch.distributions import Categorical


#def init_weights(module, gain=1.0):
#    if isinstance(module, nn.Linear):
#        nn.init.orthogonal_(module.weight, gain)
#        nn.init.constant_(module.bias, 0.0)

def init_weights(layer, gain=np.sqrt(2)):
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, envs):
        super(ActorCritic, self).__init__()

        num_features, = envs.single_observation_space.shape
        num_actions = envs.single_action_space.n

        self.network = nn.Identity()
    
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
