import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
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
        self.res_block0 = ResidualBlock(self._out_channels)
        self.res_block1 = ResidualBlock(self._out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res_block0(x)
        x = self.res_block1(x)
        assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return (self._out_channels, (h + 1) // 2, (w + 1) // 2)


class ImpalaCNN(nn.Module):
    """
    Network from IMPALA paper.
    Based on https://github.com/ray-project/ray/blob/master/rllib/models/tf/visionnet_v2.py
    and https://github.com/openai/baselines/blob/9ee399f5b20cd70ac0a871927a6cf043b478193f/baselines/common/models.py#L28
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        nn.Module.__init__(self)

        h, w, c = obs_space.shape
        shape = (c, h, w)

        conv_seqs = []
        for out_channels in [16, 32, 32]:
            conv_seq = ConvSequence(shape, out_channels)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        self.conv_seqs = nn.ModuleList(conv_seqs)
        self.hidden_fc = nn.Linear(in_features=shape[0] * shape[1] * shape[2], out_features=256)
        #self.logits_fc = nn.Linear(in_features=256, out_features=num_outputs)
        #self.value_fc = nn.Linear(in_features=256, out_features=1)

    def forward(self, obs, state):
        x = obs
        #x = x / 255.0  # scale to 0-1
        #x = x.permute(0, 3, 1, 2)  # NHWC => NCHW
        for conv_seq in self.conv_seqs:
            x = conv_seq(x)
        x = th.flatten(x, start_dim=1)
        x = F.relu(x)
        x = self.hidden_fc(x)
        x = F.relu(x)
        #logits = self.logits_fc(x)
        #value = self.value_fc(x)
        #self._value = value.squeeze(1)
        #return logits, state
        return x

    #def value(self):
    #    assert self._value is not None, "must call forward() first"
    #    return self._value