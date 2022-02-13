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

How can the problem be made harder?
How should zoom be handled? One way: objects are approximated by their center pixel far away    
"""

import gym
import numpy as np
from gym.utils import seeding
from utils import *


KEYS = [
    ord(" "),
    ord("d"),
    ord("a"),
    ord("s"),
    ord("w"),
    ord("e"),
    ord("q"),
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


class Toy(gym.Env):

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, radius):
        self.seed(0)
        self.radius = radius


        self.reward_range = (-1, 1)
        self.action_space = gym.spaces.Discrete()
        self.observation_space = gym.spaces.Box(0, 255, (*shape, 3), dtype=np.uint8)


    def step(self, action):

        a = np.array(ACTIONS[action])

        self.player += a

        py, px, pr = self.player

        edge = 

        py = clamp(py, pr, h-pr-1)
        px = clamp(py, pr, h-pr-1)
        

        return [], 0, False, {}

    def reset(self):

        self.position = (0, 0)

        h, w = self.shape
        fn=lambda x, y: 0.0
        
        self.map = np.fromfunction(fn, self.shape)

        self.player = np.array((h//2, w//2, 0))

        i = np.array([(y, x, 0) for y, x in np.index(self.map.shape)])
        w = np.array([p for p in np.flatiter(self.map)])

        self.targets = self.random.choice(i, 3, replace=False, p=softmax(w))

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
    
    def get_keys_to_action(self):
        return {(key, ): a for a, key in enumerate(KEYS)}
