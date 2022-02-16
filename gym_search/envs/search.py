import gym
import enum
import numpy as np
import cv2 as cv

from gym.utils import seeding
from gym_search.utils import clamp
from gym_search.terrain import uniform_terrain

"""
Add new targets according to Poisson distribution.
"""


class SearchEnv(gym.Env):

    metadata = {'render.modes': ['human', 'rgb_array'], }

    class Action(enum.IntEnum):
        NONE = 0
        TRIGGER = 1
        NORTH = 2
        EAST = 3
        SOUTH = 4
        WEST = 5

    def __init__(
        self, 
        world_shape=(40, 40), 
        view_shape=(5, 5), 
        step_size=1, 
        num_targets=3,
        terrain_func=uniform_terrain,
        seed=0
    ):
        self.seed(seed)
        self.world_shape = world_shape
        self.view_shape = view_shape
        self.step_size = step_size
        self.num_targets = num_targets
        self.terrain_func = terrain_func

        self.reward_range = (-1, 1)
        self.action_space = gym.spaces.Discrete(len(self.Action))
        self.observation_space = gym.spaces.Dict(dict(
            img=gym.spaces.Box(0, 255, (*self.view_shape, 3), dtype=np.uint8),
            pos=gym.spaces.Discrete(self.world_shape[0]*self.world_shape[1])
        ))

    def reset(self):
        wh, ww = self.world_shape
        vh, vw = self.view_shape
        
        self.terrain = self.terrain_func(self.world_shape, self.random)
        self.position = (self.random.randint(0, wh-vh+1), self.random.randint(0, ww-vw+1)) 

        p = self.terrain.flatten()/np.sum(self.terrain)
        t = self.random.choice(self.terrain.size, self.num_targets, replace=False, p=p)

        self.targets = [(i//ww, i%ww, False) for i in t]

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

        r = 0

        if action == self.Action.TRIGGER:

            r -= 5

            for i, t in enumerate(self.targets):
                ty, tx, hit = t

                if hit:
                    continue
                
                if px <= tx and tx < px + vw and py <= ty and ty < py + vh:
                    r += 100
                    self.targets[i] = (ty, tx, True)

        d = all(hit for _, _, hit in self.targets)

        if d:
            r += 500

        r -= 1

        return self.observe(), r, d, {}

    def render(self, mode="rgb_array", observe=False):
        if observe:
            img = self.observe()
        else:
            vh, vw = self.view_shape
            y0, x0 = self.position
            y1, x1 = y0+vh, x0+vw

            img = 1.0 - self.image(hidden=True)
            img[y0:y1,x0:x1] = 1.0 - img[y0:y1,x0:x1]
            img = img*255
            img = img.astype(dtype=np.uint8)

        if mode == "human":
            cv.imshow("search", img)
            cv.waitKey(1)
        else:
            return img


    def close(self):
        #cv.destroyWindow("search")
        #cv.destroyAllWindows()
        pass

    def seed(self, seed=None):
        self.random, _ = seeding.np_random(seed)
        return [seed]

    def observe(self):
        wh, ww = self.world_shape
        vh, vw = self.view_shape
        y0, x0 = self.position
        y1, x1 = y0+vh, x0+vw

        img = self.image(hidden=True)

        obs = img[y0:y1,x0:x1,:]
        obs = obs*255
        obs = obs.astype(dtype=np.uint8)

        return dict(
            img=obs,
            pos=y0*ww+x0
        )

    def image(self, hidden=False):
        img = np.zeros((*self.world_shape, 3))
        img[:,:,0] = self.terrain

        for ty, tx, hit in self.targets:
            if hit:
                img[ty,tx] = (0,1,0)
            elif hidden:
                img[ty,tx] = (0,0,1)

        return img

