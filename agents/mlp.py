import numpy as np
import torch as th
import torch.nn as nn
from agents.utils import init_weights


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
