"""
Map is high resolution
Camera is low resolution

Camera has a position in map
It undersamples when zoomed out but oversamples when zoomed in
The observation space determine by camera resolution

At what steps can you zoom?
Maybe oversampling should not be a thing,
but then how can you zoom in too far

Objects are usually far away



Do distractors really need to be similar at a distance?

"""

import gym
import enum
import numpy as np
import cv2 as cv
from gym.utils import seeding
from collections import defaultdict
from utils import *


KEYS = [
    ord(" "),
    ord("d"),
    ord("a"),
    ord("s"),
    ord("w"),
    ord("e"),
    ord("q"),
    ord("\n")
]

ACTIONS = [
    ( 0, 0, 0),
    ( 1, 0, 0),
    (-1, 0, 0),
    ( 0, 1, 0),
    ( 0,-1, 0),
    ( 0, 0, 1),
    ( 0, 0,-1), 
]


class Action(enum.IntEnum):
    NONE = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    IN = 5
    OUT = 6    
    TRIGGER = 7

    def key(self):
        return ord([" ", "w", "s", "a", "d", "e", "q", "\n"][self.value])

    def delta(self):
        return [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1), (0, 0), (0, 0), (0, 0)][self.value]

    def scale(self):
        return [0, 0, 0, 0, 0, 1, -1, 0][self.value]


class ToyEnv(gym.Env):

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, shape=(256, 256), view=(32, 32)):
        self.seed(0)
        self.shape = shape
        self.view = view

        self.reward_range = (-1, 1)
        self.action_space = gym.spaces.Discrete(len(Action))
        self.observation_space = gym.spaces.Box(0, 255, (*view, 3), dtype=np.uint8)

    def reset(self):
        f = lambda x, y: self.random.rand()
        g = np.vectorize(f)

        self.tiles = np.fromfunction(g, self.shape)
        self.position = (0, 0)
        self.scale = 1

        """
        i = np.array([(y, x, 0) for y, x in np.index(self.map.shape)])
        w = np.array([p for p in np.flatiter(self.map)])

        self.targets = self.random.choice(i, 3, replace=False, p=softmax(w))
        """

        return self.observe()


    def step(self, action):
        a = Action(action)

        wh, ww = self.shape
        vh, vw = self.view

        py, px = self.position
        dy, dx = a.delta()

        py = clamp(py+dy, 0, wh-1)
        px = clamp(px+dx, 0, ww-1)

        s = clamp(self.scale + a.scale(), 1, 100)
        
        print(s)

        self.position = (py, px)
        self.scale = s

        return self.observe(), 0, False, {}

    def render(self, mode="human"):
        img = self.observe()
        
        if mode == "rgb_array":
            return img
        else:
            cv.imshow("toy", img)
            cv.waitKey(0)

    def close(self):
        cv.destroyAllWindows()

    def seed(self, seed=None):
        self.random, _ = seeding.np_random(seed)
        return [seed]

    def observe(self):
        vh, vw = self.view
        y0, x0 = self.position
        y1, x1 = y0+vh, x0+vw

        obs = np.zeros((*self.view, 3), dtype=np.uint8)
        obs[:,:,0] = self.tiles[y0:y1,x0:x1]*255

        obs = cv.resize(obs, (vw*self.scale, vh*self.scale), interpolation=cv.INTER_NEAREST)
        obs = obs[:vh,:vw]

        return obs        
    
    def get_keys_to_action(self):
        return {(a.key(), ): a for a in Action}
