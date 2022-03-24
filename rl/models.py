import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import gym

from rl.utils import init_weights, preprocess_image



class MLP(nn.Module):
    def __init__(self, in_features, out_features, hidden_dims=None, out_gain=np.sqrt(2)):
        super(MLP, self).__init__()

        hidden_dims = [] if hidden_dims is None else hidden_dims
        layer_dims = [in_features] + hidden_dims + [out_features]
        layers = []

        for i in range(len(layer_dims) - 1):
            in_dim = layer_dims[i]
            out_dim = layer_dims[i+1]
            
            if i == len(layer_dims) - 2:
                layers.append(init_weights(nn.Linear(in_dim, out_dim), gain=out_gain))
            else:
                layers.append(init_weights(nn.Linear(in_dim, out_dim)))
                layers.append(nn.Tanh())

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class NatureCNN(nn.Module):
    # Mnih, Volodymyr, et al.
    # "Human-level control through deep reinforcement learning."
    
    def __init__(self, observation_space, features_dim=256):
        super().__init__()
        assert isinstance(observation_space, gym.spaces.Box)
        
        in_channels = observation_space.shape[2]

        self.convs = nn.Sequential(
            init_weights(nn.Conv2d(in_channels, 32, 8, stride=4, padding=0)),
            nn.ReLU(),
            init_weights(nn.Conv2d(32, 64, 4, stride=2, padding=0)),
            nn.ReLU(),
            init_weights(nn.Conv2d(64, 64, 3, stride=1, padding=0)),
            nn.ReLU(),
            nn.Flatten(),
        )

        with th.no_grad():
            hidden_dim = self.convs(preprocess_image(th.tensor(observation_space.sample()).unsqueeze(0).float())).shape[1]

        self.linear = nn.Sequential(
            init_weights(nn.Linear(hidden_dim, features_dim)),
            nn.ReLU()
        )

        self.features_dim = features_dim

    def forward(self, obs):
        return self.linear(self.convs(obs))

    def _output_shape(self, input_shape, kernel_size, stride=1, padding=0):
        return (np.array(input_shape) - kernel_size+ 2*padding)//stride + 1



class AlphaCNN(nn.Module):
    """
    Network from AlphaZero paper (with modifications).
    """

    class ResidualBlock(nn.Module):
        def __init__(self, filters):
            super().__init__()

            self.conv1 = init_weights(nn.Conv2d(filters, filters, 3, padding=1))
            self.norm1 = nn.BatchNorm2d(filters)
            self.conv2 = init_weights(nn.Conv2d(filters, filters, 3, padding=1))
            self.norm2 = nn.BatchNorm2d(filters)
        
            # todo: init weights?

        def forward(self, x):
            y = x

            x = self.conv1(x)
            x = self.norm1(x)
            x = F.relu(x)

            x = self.conv2(x)
            x = self.norm2(x)
            x = F.relu(x)

            x = x + y
            x = F.relu(x)

            return x

    def __init__(self, observation_space, features_dim=256, filters=64, blocks=2):
        super().__init__()
        
        assert isinstance(observation_space, gym.spaces.Box)

        channels = observation_space.shape[2]
        
        self.initial = nn.Sequential(
            init_weights(nn.Conv2d(channels, filters, 3, stride=1, padding=1)),
            nn.BatchNorm2d(filters),
            nn.ReLU()
        )

        self.residual = nn.Sequential(
            *[self.ResidualBlock(filters) for _ in range(blocks)],
            nn.Flatten(),
        )

        with th.no_grad():
            hidden_dim = self.residual(self.initial(preprocess_image(th.tensor(observation_space.sample()).unsqueeze(0).float()))).shape[1]

        self.linear = nn.Sequential(
            init_weights(nn.Linear(hidden_dim, features_dim)),
            nn.ReLU()
        )

        self.features_dim = features_dim
        self.hidden_dim = hidden_dim
        self.channels = channels
        self.filters = filters
        self.blocks = blocks
    
    def forward(self, obs):
        x = self.initial(obs)
        x = self.residual(x)
        x = self.linear(x)

        return x


class ImpalaCNN(nn.Module):
    """
    Network from IMPALA paper.
    """

    class ResidualBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.conv0 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
            self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        
        def forward(self, x):
            y = x
            x = F.relu(x)
            x = self.conv0(x)
            x = F.relu(x)
            x = self.conv1(x)
            return x + y

    class ConvSequence(nn.Module):
        def __init__(self, input_shape, out_channels):
            super().__init__()
            self.input_shape = input_shape
            self.out_channels = out_channels

            self.conv = nn.Conv2d(in_channels=self.input_shape[0], out_channels=self.out_channels, kernel_size=3, padding=1)
            
            self.res_block0 = ImpalaCNN.ResidualBlock(self.out_channels)
            self.res_block1 = ImpalaCNN.ResidualBlock(self.out_channels)

        def forward(self, x):
            x = self.conv(x)
            x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
            x = self.res_block0(x)
            x = self.res_block1(x)
            assert x.shape[1:] == self.get_output_shape()
            return x

        def get_output_shape(self):
            _c, h, w = self.input_shape
            return (self.out_channels, (h + 1) // 2, (w + 1) // 2)

    def __init__(self, obs_space, features_dim=256):
        super().__init__()

        h, w, c = obs_space.shape
        shape = (c, h, w)

        convs = []
        
        for out_channels in [16, 32, 32]:
            conv_seq = ImpalaCNN.ConvSequence(shape, out_channels)
            shape = conv_seq.get_output_shape()
            convs.append(conv_seq)

        self.convs = nn.Sequential(
            *convs,
            nn.Flatten(),
            nn.ReLU(),
        )

        self.linear = nn.Sequential(
            nn.Linear(shape[0]*shape[1]*shape[2], features_dim),
            nn.ReLU()
        )

        self.features_dim = features_dim


    def forward(self, obs):
        x = obs
        x = self.convs(x)
        x = self.linear(x)

        return x