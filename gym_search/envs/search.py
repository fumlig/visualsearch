import gym
import enum
import numpy as np

from collections import defaultdict


class Action(enum.IntEnum):
    NONE = 0
    TRIGGER = 1
    UP = 2
    RIGHT = 3
    DOWN = 4
    LEFT = 5


class SearchEnv(gym.Env):

    metadata = {"render.modes": ["rgb_array"]}

    def __init__(
        self,
        shape,
        view,
        wrap=False,
        train_steps=1000,
        test_steps=5000,
        train_samples=None,
        test_samples=1000,
    ):
        self.shape = shape
        self.view = view
        self.wrap = wrap

        self.train_steps = train_steps
        self.test_steps = test_steps
        self.train_samples = train_samples if train_samples is not None else np.iinfo(np.int64).max - test_samples
        self.test_samples = test_samples

        self.training = True

        self.reward_range = (-np.inf, np.inf)
        self.action_space = gym.spaces.Discrete(len(self.Action))
        self.observation_space = gym.spaces.Dict(dict(image=gym.spaces.Box(0, 255, (*self.view, 3), dtype=np.uint8), position=gym.spaces.MultiDiscrete(self.shape)))


    def reset(self, seed=None):
        if seed is not None:
            self.np_random, seed = gym.utils.seeding.np_random(seed)

        if self.training:
            targets = self.generate(self.np_random.integers(self.test_samples, self.test_samples + self.train_samples))
        else:
            targets = self.generate(self.np_random.integers(0, self.test_samples))

        self.position = np.array([self.np_random.integers(0, d) for d in self.shape])
        self.targets = targets
        self.hits = [False for _ in range(len(self.targets))]
        self.path = [self.position]
        self.num_steps = 0
        self.counters = defaultdict(int)

        return self.observation()

    def step(self, action):

        if action != Action.TRIGGER:
            step = self.get_action_step(action)
            self.position += step

            if self.wrap:
                self.position %= self.shape
            else:
                self.position = np.clip(self.position, (0, 0), self.shape)
        else:
            hits = 0

            for i in range(len(self.targets)):
                if self.hits[i]:
                    continue

                if self.visible(self.targets[i]):
                    self.hits[i] = True
                    hits += 1

        self.num_steps += 1

        obs = self.observation()
        rew = hits*10 if hits else -1
        done = all(self.hits) or self.num_steps >= self.max_steps
        info = {
            "position": self.position,
            "targets": self.targets,
            "hits": self.hits,
            "path": self.path,
            "success": all(self.hits)
        }

        return obs, rew, done, info

    def render(self, mode="rgb_array"):
        raise NotImplementedError

    def observation(self):
        raise NotImplementedError

    def visible(self, target):
        raise NotImplementedError

    def sample(self, seed):
        raise NotImplementedError


    def train(self, mode=True):
        self.training = mode
    
    def test(self):
        self.train(False)


    def get_action_meanings(self):
        return [a.name for a in Action]
    
    def get_keys_to_action(self):
        return {
            (ord(" "),): Action.TRIGGER,
            (ord("w"),): Action.UP,
            (ord("d"),): Action.RIGHT,
            (ord("s"),): Action.DOWN,
            (ord("a"),): Action.LEFT,
        }

    def get_action_step(self, action):
        return {
            Action.UP:      (-1, 0),
            Action.RIGHT:   ( 0, 1),
            Action.DOWN:    ( 1, 0),
            Action.LEFT:    ( 0,-1)
        }.get(action, (0, 0))