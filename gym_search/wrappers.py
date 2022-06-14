import numpy as np
import cv2 as cv
import gym

from gym_search.utils import to_index

class ObserveTime(gym.ObservationWrapper):
    def __init__(self, env, key="time"):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Dict)
        
        self.key = key

        self.observation_space = gym.spaces.Dict()

        for key, space in self.env.observation_space.items():
            self.observation_space[key] = space
        
        self.observation_space[self.key] = gym.spaces.Discrete(self.max_steps)

    def observation(self, observation):
        obs = observation.copy()
        obs[self.key] = self.num_steps
        return obs


class ResizeImage(gym.ObservationWrapper):
    def __init__(self, env, key="image", size=(64, 64)):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Dict)
        
        self.key = key
        self.size = size

        self.observation_space = gym.spaces.Dict()

        for key, space in self.env.observation_space.items():
            self.observation_space[key] = space
        
        self.observation_space[self.key] = gym.spaces.Box(0, 255, self.size + self.env.observation_space[self.key].shape[2:], dtype=np.uint8)

    def observation(self, observation):
        obs = observation.copy()
        obs[self.key] = cv.resize(obs[self.key], self.size, interpolation=cv.INTER_AREA)
        return obs


class LastAction(gym.ObservationWrapper):
    def __init__(self, env, key="last_action"):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Dict)
        
        self.key = key

        self.observation_space = gym.spaces.Dict()

        for key, space in self.env.observation_space.items():
            self.observation_space[key] = space
        
        self.observation_space[self.key] = gym.spaces.Discrete(self.env.action_space.n)
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self.observation(obs)
        self.last_action = action
        return obs, reward, done, info

    def reset(self, **kwargs):
        self.last_action = 0
        if kwargs.get("return_info", False):
            obs, info = self.env.reset(**kwargs)
            return self.observation(obs), info
        else:
            return self.observation(self.env.reset(**kwargs))

    def observation(self, observation):
        obs = observation.copy()
        obs[self.key] = self.last_action
        return obs


class LastReward(gym.ObservationWrapper):
    def __init__(self, env, key="last_reward"):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Dict)
        
        self.key = key

        self.observation_space = gym.spaces.Dict()

        for key, space in self.env.observation_space.items():
            self.observation_space[key] = space
        
        self.observation_space[self.key] = gym.spaces.Box(*self.env.reward_range, (1,))
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self.observation(obs)
        self.last_reward = np.array([reward])
        return obs, reward, done, info

    def reset(self, **kwargs):
        self.last_reward = np.array([0.0])
        if kwargs.get("return_info", False):
            obs, info = self.env.reset(**kwargs)
            return self.observation(obs), info
        else:
            return self.observation(self.env.reset(**kwargs))

    def observation(self, obs):
        new_obs = obs.copy()
        new_obs["last_reward"] = self.last_reward
        return new_obs