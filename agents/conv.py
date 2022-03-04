import numpy as np
import gym


def is_image_space(obs_space):
    return isinstance(obs_space, gym.spaces.Box) and \
        len(obs_space.shape == 3) and obs_space.shape[2] == 3 and obs_space.dtype == np.uint8


class ChannelWrapper(gym.ObservationWrapper):
