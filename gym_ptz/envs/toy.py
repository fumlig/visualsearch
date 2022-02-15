import gym
import enum
import numpy as np
import cv2 as cv
from gym.utils import seeding
from gym_ptz.utils import clamp, softmax, perlin
import PIL


class ToyEnv(gym.Env):

    metadata = {'render.modes': ['rgb_array']}

    class Action(enum.IntEnum):
        NONE = 0
        TRIGGER = 1
        NORTH = 2
        EAST = 3
        SOUTH = 4
        WEST = 5

    def __init__(self, world_shape=(64, 64), view_shape=(8, 8), step_size=8, num_targets=5):
        self.seed(0)
        self.world_shape = world_shape
        self.view_shape = view_shape
        self.step_size = step_size
        self.num_targets = num_targets

        self.reward_range = (-1, 1)
        self.action_space = gym.spaces.Discrete(len(self.Action))
        self.observation_space = gym.spaces.Box(0, 255, (*self.view_shape, 3), dtype=np.uint8)

    def reset(self):
        #f = lambda x, y: self.random.rand()
        #g = np.vectorize(f)

        #f = lambda x, y: (1+perlin(0.1*x, 0.1*y, random=self.random))/2
        #g = np.vectorize(f)

        g = lambda x, y: (1 + perlin(x*0.01, y*0.01, random=self.random))/2

        self.tiles = np.fromfunction(g, self.world_shape)
        self.position = (0, 0)

        w = np.array([p for p in self.tiles.flat])
        p = softmax(w)
        t = self.random.choice(self.tiles.size, self.num_targets, replace=False, p=p)

        wh, wv = self.world_shape
        self.targets = [(i//wv, i%wv, False) for i in t]

        return self.observe()


    def step(self, action):
        wh, ww = self.world_shape
        vh, vw = self.view_shape
        py, px = self.position
        dy, dx = {
            self.Action.NORTH:   (-1, 0),
            self.Action.EAST:    ( 0, 1),
            self.Action.SOUTH:   ( 1, 0),
            self.Action.WEST:    ( 0,-1)
        }.get(action, (0, 0))

        py = clamp(py+dy*self.step_size, 0, wh-vh)
        px = clamp(px+dx*self.step_size, 0, ww-vw)

        self.position = (py, px)

        r = -1

        if action == self.Action.TRIGGER:
            for i, t in enumerate(self.targets):
                ty, tx, hit = t

                if hit:
                    continue
                
                if px <= tx and tx < px + vw and py <= ty and ty < py + vh:
                    r = 1
                    self.targets[i] = (ty, tx, True)

        d = all(hit for _, _, hit in self.targets)

        return self.observe(), r, d, {}

    def render(self, mode="rgb_array", observe=False):
        if observe:
            img = self.observe()
        else:
            vh, vw = self.view_shape
            y0, x0 = self.position
            y1, x1 = y0+vh, x0+vw

            img = self.image(hidden=True)
            darken = img.copy()*0.5
            img[:,:,0] -= darken[:,:,0]
            img[y0:y1,x0:x1,0] += darken[y0:y1,x0:x1,0]
            img = img*255
            img = img.astype(dtype=np.uint8)

        return img


    def close(self):
        cv.destroyAllWindows()

    def seed(self, seed=None):
        self.random, _ = seeding.np_random(seed)
        return [seed]

    def observe(self):
        vh, vw = self.view_shape
        y0, x0 = self.position
        y1, x1 = y0+vh, x0+vw

        img = self.image(hidden=True)

        obs = img[y0:y1,x0:x1,:]
        obs = obs*255
        obs = obs.astype(dtype=np.uint8)

        return obs

    def image(self, hidden=False):
        img = np.zeros((*self.world_shape, 3))
        img[:,:,0] = self.tiles

        for ty, tx, hit in self.targets:
            if hit:
                img[ty,tx] = (0,1,0)
            elif hidden:
                img[ty,tx] = (0,0,1)

        return img

