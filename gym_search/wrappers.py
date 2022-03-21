import numpy as np
import cv2 as cv
import gym


class TimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        super().__init__(env)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        
        if self.env.spec is not None:
            self.env.spec.max_episode_steps = max_episode_steps
        
        self.max_episode_steps = max_episode_steps
        self.num_steps = None

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.num_steps += 1
        if self.num_steps >= self.max_episode_steps:
            info["TimeLimit.truncated"] = not done
            done = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self.num_steps = 0
        return self.env.reset(**kwargs)


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

class ResizeImage(InsertObservation):
    def __init__(self, env, key="image", shape=(64, 64)):
        super().__init__(
            env,
            key,
            lambda self=self: gym.spaces.Box(0, 255, shape + self.env.observation_space[key].shape[2:], dtype=np.uint8),
            lambda self=self: cv.resize(self.env.observation()[key], shape, interpolation=cv.INTER_AREA)
        )


class ObservePosition(InsertObservation):
    def __init__(self, env, key="position"):
        super().__init__(
            env,
            key,
            lambda self=self: gym.spaces.Discrete(self.shape[0]//self.view.shape[0] * self.shape[1]//self.view.shape[1]),
            lambda self=self: self.view.pos[0]//self.view.shape[0]*self.shape[1]//self.view.shape[1]+self.view.pos[1]//self.view.shape[1]
        )


class ObserveTime(InsertObservation):
    def __init__(self, env, key="time"):
        super().__init__(
            env,
            key,
            lambda self=self: gym.spaces.Discrete(self.max_steps),
            lambda self=self: self.num_steps
        )


class ObserveVisible(InsertObservation):
    def __init__(self, env, key="visible"):
        super().__init__(
            env,
            key,
            lambda self=self: gym.spaces.Box(0, 1, self.shape),
            lambda self=self: self.visible
        )


class ObserveVisited(InsertObservation):
    def __init__(self, env, key="visited"):
        super().__init__(
            env,
            key,
            lambda self=self: gym.spaces.Box(0, 1, self.shape),
            lambda self=self: self.visited
        )

class ObserveTriggered(InsertObservation):
    def __init__(self, env, key="triggered"):
        super().__init__(
            env,
            key,
            lambda self=self: gym.spaces.Box(0, 1, self.shape),
            lambda self=self: self.triggered
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