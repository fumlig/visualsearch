"""
Tiles with different integers
Laid out in some pattern
Targets laid out with probability related to pattern
Distractors placed randomly
Different shape from target

Triggering all of the time should not be allowed: give penalty
Targets should be found quickly: give time penalty
Targets should be found accurately: give reward based on overlap
Random number of targets? How can it know how many there are? - It does not need to, the episode ends when it is done. It should just keep looking
"""

import gym
import numpy as np
from gym.utils import seeding
from utils import *


def place(map, n):
    pass


class Toy(gym.Env):

    metadata = {'render.modes': ['human', 'rgb_array']}

    NUM_TARGETS = 3

    def __init__(self, size):
        self.seed(0)
        self.size = size

        self.reward_range = (-1, 1)
        self.action_space = gym.spaces.Discrete()
        self.observation_space = gym.spaces.Box(0, 255, (size, size, 3), dtype=np.uint8)


    def step(self, action):
        

        py0 = clamp(py-pr, pr, h-pr-1)
        py1 = clamp(py+pr, pr, h-pr-1)
        px0 = clamp(px-pr, pr, w-pr-1)
        px1 = clamp(px+pr, pr, w-pr-1)

        return [], 0, False, {}

    def reset(self):
        """
        initialize some map of weights
        use this map of weights to randomize
        - environment
        - targets
        - distractors

        how the fuck is this supposed to be generalizable?
        it is not. but properties of each separate environment can
        be understood. the map represents the properties of the environment

        example: map represents probability of enemies
        colors is set depending on probability
        """

        h, w = self.shape
        fn=lambda x, y: 0.0
        
        self.map = np.fromfunction(fn, self.shape)

        self.player = np.array((h//2, w//2, 0))

        i = np.array([(y, x, 0) for y, x in np.index(self.map.shape)])
        w = np.array([p for p in np.flatiter(self.map)])
    
        self.targets = self.random.choice(i, self.NUM_TARGETS, replace=False, p=softmax(w))

        return self.observe()

    def render(self, mode="human"):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        self.random, _ = seeding.np_random(seed)
        return [seed]

    def observe(self):
        img = self.image()
        h, w = self.shape

        py, px, pr = self.player
        py0 = py-pr
        py1 = py+pr
        px0 = px-pr
        px1 = px+pr

        obs = img[py0:py1,px0:px1]

        return obs


    def image(self):
        h, w = self.shape
        img = np.zeros((*self.shape, 3), dtype=np.uint8)

        img[:,:,2] = self.map*255

        for ty, tx, tr in self.targets:
            ty0 = clamp(ty-tr, 0, h-1)
            ty1 = clamp(ty+tr, 0, h-1)
            tx0 = clamp(tx-tr, 0, w-1)
            tx1 = clamp(tx+tr, 0, w-1)
            
            img[ty0:ty1,tx0:tx1] = (255, 0, 0)

        return img
    