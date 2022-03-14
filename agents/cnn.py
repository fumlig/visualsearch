import numpy as np
import torch as th
import torch.nn as nn
import gym

from agents.utils import init_weights, preprocess_image


class NatureCNN(nn.Module):
    # Mnih, Volodymyr, et al.
    # "Human-level control through deep reinforcement learning."
    
    def __init__(self, observation_space, features_dim=512):
        super(NatureCNN, self).__init__()
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
            hidden_dim = self.cnn(preprocess_image(th.tensor(observation_space.sample()).float())).shape[1]

        self.linear = nn.Sequential(
            init_weights(nn.Linear(hidden_dim, features_dim)),
            nn.ReLU()
        )

        self.features_dim = features_dim

    def forward(self, obs):
        return self.linear(self.cnn(obs))

    def _output_shape(self, input_shape, kernel_size, stride=1, padding=0):
        return (np.array(input_shape) - kernel_size+ 2*padding)//stride + 1
