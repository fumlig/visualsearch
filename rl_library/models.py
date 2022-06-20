import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import gym

from rl.utils import init_weights, preprocess_image


class MLP(nn.Module):
    """Multi-Layer Perceptron."""

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
                layers.append(nn.ReLU())

        self.output_dim = out_features
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class NatureCNN(nn.Module):
    """Human-level control through deep reinforcement learning" (Mnih et al., 2015)"""
    
    def __init__(self, observation_space, output_dim=512):
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
            init_weights(nn.Linear(hidden_dim, output_dim)),
            nn.ReLU()
        )

        self.output_dim = output_dim

    def forward(self, obs):
        x = self.convs(obs)
        x = self.linear(x)
        return x

    def _output_shape(self, input_shape, kernel_size, stride=1, padding=0):
        return (np.array(input_shape) - kernel_size+ 2*padding)//stride + 1


class ImpalaCNN(nn.Module):
    """
    Network from IMPALA paper.
    """

    class ResidualBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.conv0 = init_weights(nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1))
            self.conv1 = init_weights(nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1))
        
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

            self.conv = init_weights(nn.Conv2d(in_channels=self.input_shape[0], out_channels=self.out_channels, kernel_size=3, padding=1))
            
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

    def __init__(self, obs_shape, output_dim=256):
        super().__init__()

        h, w, c = obs_shape
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
            init_weights(nn.Linear(shape[0]*shape[1]*shape[2], output_dim)),
            nn.ReLU()
        )

        self.output_dim = output_dim


    def forward(self, obs):
        x = obs
        x = self.convs(x)
        x = self.linear(x)

        return x


class NeuralMap(nn.Module):
    """
    Implementation of Neural Map by Parisotto et al. (https://arxiv.org/abs/1702.08360)
    """

    class MapCNN(nn.Module):

        def __init__(self, observation_shape, output_dim=256):
            super().__init__()
            
            in_channels = observation_shape[2]
            hidden_channels = 32

            self.convs = nn.Sequential(
                init_weights(nn.Conv2d(in_channels, hidden_channels, 3, padding=1)),
                nn.ReLU(),
                init_weights(nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1)),
                nn.ReLU(),
                init_weights(nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1)),
                nn.ReLU(),
                nn.Flatten(),
            )

            hidden_dim = np.prod((*observation_shape[:2], hidden_channels))

            self.linear = nn.Sequential(
                init_weights(nn.Linear(hidden_dim, output_dim)),
                nn.ReLU(),
            )

            self.output_dim = output_dim

        def forward(self, obs):
            return self.linear(self.convs(obs))


    def __init__(self, map_shape, input_dim, features_dim=64):
        super().__init__()
        # all of the output the same number of features, I think...
        self.read_net = self.MapCNN((*map_shape, features_dim), features_dim)
        self.query_net = MLP(input_dim + features_dim, features_dim)
        self.write_net = MLP(input_dim + features_dim + features_dim + features_dim, features_dim)
        
        self.features_dim = features_dim
        self.input_dim = input_dim
        self.output_dim = features_dim + features_dim + features_dim
        self.shape = (features_dim, *map_shape)

    def read(self, state):
        return self.read_net(state)

    def context(self, x, r, state):
        b = state.shape[0]
        c = self.features_dim
        q = self.query_net(th.cat([x, r], dim=1))
        s = state.view(b, c, -1)
        a = th.bmm(q.view(b, -1, c), s)
        a = F.softmax(a, dim=2)
        c = th.sum(a*s, dim=2)
        return c

    def write(self, x, r, c, state, index):
        b = x.shape[0]
        m = state[th.arange(b),:,index[:,0],index[:,1]]
        w = self.write_net(th.cat([x, r, c, m], dim=1))
        return w

    def forward(self, x, state, index):
        r = self.read(state)
        c = self.context(x, r, state)
        w = self.write(x, r, c, state, index)

        b = x.shape[0]
        new_state = state.clone()
        new_state[th.arange(b),:,index[:,0],index[:,1]] = w

        return th.cat([r, c, w], dim=1), new_state


class Map(nn.Module):
    """
    Modified implementation of NeuralMap.
    
    Simplified for search environments (no context read).
    """

    class MapCNN(nn.Module):

        def __init__(self, observation_shape, output_dim=256):
            super().__init__()
            
            in_channels = observation_shape[2]
            hidden_channels = 32

            self.convs = nn.Sequential(
                init_weights(nn.Conv2d(in_channels, hidden_channels, 3, padding=1)),
                nn.ReLU(),
                init_weights(nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1)),
                nn.ReLU(),
                init_weights(nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1)),
                nn.ReLU(),
                nn.Flatten(),
            )

            hidden_dim = np.prod((*observation_shape[:2], hidden_channels))

            self.linear = nn.Sequential(
                init_weights(nn.Linear(hidden_dim, output_dim)),
                nn.ReLU()
            )

            self.output_dim = output_dim

        def forward(self, obs):
            return self.linear(self.convs(obs))


    def __init__(self, map_shape, input_dim, features_dim=64):
        super().__init__()
        self.read_net = self.MapCNN((*map_shape, features_dim), features_dim)
        self.write_net = MLP(input_dim + features_dim + features_dim, features_dim)
        
        self.features_dim = features_dim
        self.input_dim = input_dim
        self.output_dim = features_dim + features_dim
        self.shape = (features_dim, *map_shape)

    def read(self, state):
        return self.read_net(state)

    def write(self, x, r, state, index):
        b = x.shape[0]
        m = state[th.arange(b),:,index[:,0],index[:,1]]
        w = self.write_net(th.cat([x, r, m], dim=1))
        return w

    def forward(self, x, state, index):
        r = self.read(state)
        w = self.write(x, r, state, index)

        b = x.shape[0]

        new_state = state.clone()
        new_state[th.arange(b),:,index[:,0],index[:,1]] = w

        return th.cat([r, w], dim=1), new_state

