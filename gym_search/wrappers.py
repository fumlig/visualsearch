import numpy as np
import cv2 as cv
import gym


class InsertObservation(gym.ObservationWrapper):
    def __init__(self, env, key, space_func, value_func):
        super().__init__(env)

        assert isinstance(env.observation_space, gym.spaces.Dict)

        self.key = key
        self.space_func = space_func
        self.value_func = value_func
        self.observation_space = gym.spaces.Dict()

        for key, space in self.env.observation_space.items():
            self.observation_space[key] = space
        
        self.observation_space[self.key] = self.space_func()

    def observation(self, obs):
        new_obs = obs.copy()
        new_obs.update({self.key: self.value_func()})
        return new_obs

class ObserveTime(InsertObservation):
    def __init__(self, env, key="time"):
        super().__init__(
            env,
            key,
            lambda self=self: gym.spaces.Discrete(self.max_steps),
            lambda self=self: self.num_steps
        )


class ExplicitMemory(InsertObservation):
    def __init__(self, env, key="memory"):
        super().__init__(
            env,
            key,
            lambda self=self: gym.spaces.Box(0, 1, (*self.scaled_shape, 3)),
            lambda self=self: np.stack([
                self.visible,
                self.visited,
                self.triggered
            ], axis=-1)
        )

class ResizeImage(InsertObservation):
    def __init__(self, env, key="image", shape=(64, 64)):
        super().__init__(
            env,
            key,
            lambda self=self: gym.spaces.Box(0, 255, shape + self.env.observation_space[key].shape[2:], dtype=np.uint8),
            lambda self=self: cv.resize(self.env.observation()[key], shape, interpolation=cv.INTER_AREA)
        )

class LastAction(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Dict)
        
        self.observation_space = gym.spaces.Dict()

        for key, space in self.env.observation_space.items():
            self.observation_space[key] = space
        
        self.observation_space["last_action"] = gym.spaces.Discrete(self.env.action_space.n)
    
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

    def observation(self, obs):
        new_obs = obs.copy()
        new_obs["last_action"] = self.last_action
        return new_obs

class LastReward(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Dict)
        
        self.observation_space = gym.spaces.Dict()

        for key, space in self.env.observation_space.items():
            self.observation_space[key] = space
        
        self.observation_space["last_reward"] = gym.spaces.Box(*self.env.reward_range, (1,))
    
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