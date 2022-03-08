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


def init_weights(layer, gain=np.sqrt(2)):
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


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



class CNN(nn.Module):
    # "NatureCNN"
    # Mnih, Volodymyr, et al.
    #"Human-level control through deep reinforcement learning."
    
    def __init__(self, observation_space, features_dim=512):
        super(CNN, self).__init__()
        assert isinstance(observation_space, gym.spaces.Box)
        
        in_channels = observation_space.shape[2]

        self.cnn = nn.Sequential(
            init_weights(nn.Conv2d(in_channels, 32, 8, stride=4, padding=0)),
            nn.ReLU(),
            init_weights(nn.Conv2d(32, 64, 4, stride=2, padding=0)),
            nn.ReLU(),
            init_weights(nn.Conv2d(64, 64, 3, stride=1, padding=0)),
            nn.ReLU(),
            nn.Flatten(),
        )

        with th.no_grad():
            hidden_dim = self.cnn(th.tensor(observation_space.sample()[np.newaxis]).float().permute(0, 3, 1, 2)).shape[1]

        self.linear = nn.Sequential(
            init_weights(nn.Linear(hidden_dim, features_dim)),
            nn.ReLU()
        )

        self.features_dim = features_dim

    def forward(self, obs):
        return self.linear(self.cnn(obs.permute(0, 3, 1, 2)))

    def _output_shape(self, input_shape, kernel_size, stride=1, padding=0):
        return (np.array(input_shape) - kernel_size+ 2*padding)//stride + 1



class Extractor(nn.Module):
    def __init__(self, observation_space):
        super(Extractor, self).__init__()
        assert isinstance(observation_space, gym.spaces.Dict)

        extractors = {}
        features_dim = 0

        for key, space in observation_space.items():
            if isinstance(space, gym.spaces.Box):
                if key == "image":
                    extractors[key] = CNN(space)
                    features_dim += extractors[key].features_dim
                else:
                    extractors[key] = nn.Flatten()
                    self.features_dim += gym.spaces.flatdim(space)
            elif isinstance(space, gym.spaces.Discrete):
                # discrete should be made one hot...
                extractors[key] = lambda x: F.one_hot(x, num_classes=space.n)
                features_dim += gym.spaces.flatdim(space)
            else:
                assert False

        self.extractors = nn.ModuleDict(extractors)
        self.features_dim = features_dim

    def forward(self, obs):
        tensors = []

        for key, observation in obs.items():
            x = self.extractors[key](observation)
            tensors.append(x)
        
        return th.cat(tensors, dim=1)


class ActorCritic(nn.Module):
    def __init__(self, envs):
        super(ActorCritic, self).__init__()

        self.extractor = Extractor(envs.single_observation_space)
        self.network = nn.Identity()

        self.policy = Actor(self.extractor.features_dim, envs.single_action_space.n, hidden_dims=[64, 64])
        self.value = Critic(self.extractor.features_dim, hidden_dims=[64, 64])

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

"""
class ConvActorCritic(nn.Module):
    def __init__(self, envs):
        super(ConvActorCritic, self).__init__()

        in_height, in_width, in_channels = envs.single_observation_space["image"].shape
        num_actions = envs.single_action_space.n
        num_features = 256
        
        out_height, out_width = conv_output_shape(
            conv_output_shape(
                conv_output_shape(
                    (in_height, in_width), 8, stride=4
                ),4, stride=2
            ), 3, stride=1
        )

        self.network = nn.Sequential(
            init_weights(nn.Conv2d(in_channels, 32, 8, stride=4)),
            nn.ReLU(),
            init_weights(nn.Conv2d(32, 32, 4, stride=2)),
            nn.ReLU(),
            init_weights(nn.Conv2d(32, 32, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            init_weights(nn.Linear(out_height*out_width*32, num_features)),
            nn.ReLU(),
        )

        self.policy = Actor(num_features, num_actions)
        self.value = Critic(num_features)

        #self.network.apply(lambda m: init_weights(m, np.sqrt(2)))
        #self.actor.apply(lambda m: init_weights(m, 0.01))
        #self.critic.apply(lambda m: init_weights(m, 1.0))

    def forward(self, obs):
        x = obs["image"].permute(0, 3, 1, 2) # n, h, w, c -> n, c, h, w
        h = self.network(x)
        pi = self.policy(h)
        v = self.value(h)
        return pi, v

    def predict(self, obs):
        x = obs["image"].permute(0, 3, 1, 2) # n, h, w, c -> n, c, h, w
        h = self.network(x)
        pi = self.policy(h)
        return pi.sample().item()
"""