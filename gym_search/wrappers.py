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

        for key, space in env.observation_space.items():
            self.observation_space[key] = space
        
        self.observation_space[self.key] = self.space_func(self.env)

    def observation(self, obs):
        new_obs = obs.copy()
        new_obs.update({self.key: self.value_func(self.env)})
        return new_obs


class InsertPosition(InsertObservation):
    def __init__(self, env):
        super().__init__(
            env,
            "position",
            lambda env: gym.spaces.Discrete(env.shape[0]//env.view.shape[0] * env.shape[1]//env.view.shape[1]),
            lambda env: env.view.pos[0]//env.view.shape[0]*env.shape[1]//env.view.shape[1]+env.view.pos[1]//env.view.shape[1]
        )