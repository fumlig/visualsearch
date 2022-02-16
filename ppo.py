import numpy as np
import torch as th
import torch.nn as nn

import gym.spaces

"""
Common:
feature extractor: obs -> tensor
network architecture: fully connected to action or value

"""


def feature_extractor(obs_space):
    if isinstance(obs_space, gym.spaces.Box):
        pass
    elif isinstance(obs_space, gym.spaces.Discrete):
        pass
    elif isinstance(obs_space, gym.spaces.Dict):
        pass


class DictExtractor(nn.Module):

    def __init__(self, observation_space: gym.spaces.Dict):
        # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
        super(CombinedExtractor, self).__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
                # The observation key is a vector, flatten it if needed
                extractors[key] = nn.Flatten()
                total_concat_size += gym.spaces.utils.flatdim(subspace)

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return th.cat(encoded_tensor_list, dim=1)




def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class PPOAgent(nn.Module):
    def __init__(self, envs):
        super(PPOAgent, self).__init__()

        input_dim = gym.spaces.flatdim(envs.single_observation_space)



        self.critic = nn.Sequential(
            layer_init(nn.)
        )



if __name__ == "__main__":
    import gym_search

    envs = gym.vector.make("Search-v0", 8, asynchronous=False)
    agent = PPOAgent(envs)
