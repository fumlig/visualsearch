import gym
import enum
import numpy as np


class BaseEnv(gym.Env):

    metadata = {"render.modes": ["rgb_array"]}

    class Action(enum.IntEnum):
        NONE = 0
        TRIGGER = 1
        NORTH = 2
        EAST = 3
        SOUTH = 4
        WEST = 5

    def __init__(
        self,
        shape,
        view,
        generator,
        wrap=False,
        train_steps=1000,
        test_steps=5000,
        train_samples=None,
        test_samples=1000,
    ):
        self.shape = shape
        self.view = view
        self.wrap = wrap
        self.generator = generator()

        self.train_steps = train_steps
        self.test_steps = test_steps
        self.train_samples = train_samples
        self.test_samples = test_samples

        self.training = True

        self.reward_range = (-np.inf, np.inf)
        self.action_space = gym.spaces.Discrete(len(self.Action))
        self.observation_space = gym.spaces.Dict(dict(image=gym.spaces.Box(0, 255, (*self.view, 3), dtype=np.uint8), position=gym.spaces.MultiDiscrete(self.shape)))


    def reset(self, seed=None):
        if seed is not None:
            self.np_random, seed = gym.utils.seeding.np_random(seed)

        self.position = np.array([self.np_random.integers(0, d) for d in self.shape])



        self.scene = self.generator.sample()


    def step(self, action):
        pass

    def render(self, mode="rgb_array"):
        raise NotImplementedError

    def close(self):
        pass

    def train(self, mode=True):
        self.training = mode
    
    def test(self):
        self.train(False)
