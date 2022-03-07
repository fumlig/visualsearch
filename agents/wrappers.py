import numpy as np
import gym


class StackChannels(gym.ObservationWrapper):

    def __init__(self, env):
        super(StackChannels, self).__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Dict)
        
        height = width = depth = 0

        for _key, space in env.observation_space.items():
            if isinstance(space, gym.spaces.Discrete):
                depth += 1
            elif isinstance(space, gym.spaces.Box):
                if len(space.shape) == 2 or len(space.shape) == 3:
                    height = max(height, space.shape[0])
                    width = max(width, space.shape[1])
                    depth += space.shape[2]
            else:
                assert False

        self.shape = (depth, height, width)

    def observation(self, obs):
        new_obs = np.zeros(self.shape)
        channel = 0

        for _key, sub_obs in obs.items():
            sub_obs = np.array(sub_obs)

            if len(sub_obs.shape) == 0:
                new_obs[:,:,channel] = sub_obs
                channel += 1
            else:
                new_obs[:sub_obs.shape[0], :sub_obs.shape[1], channel:channel+sub_obs.shape[2]] = sub_obs
                channel += sub_obs.shape[2]