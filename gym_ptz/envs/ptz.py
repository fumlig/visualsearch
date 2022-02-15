import gym
import enum
import numpy as np
import cv2 as cv
from gym.utils import seeding
from gym_ptz.utils import clamp, softmax, perlin, gaussian_kernel


class PTZEnv(gym.Env):

    metadata = {'render.modes': ['rgb_array']}

    class Action(enum.IntEnum):
        NONE = 0
        TRIGGER = 1
        NORTH = 2
        EAST = 3
        SOUTH = 4
        WEST = 5

    def __init__(self, world_shape=(32, 32), view_shape=(4, 4), step_size=1, num_targets=3):
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

        #g = lambda x, y: (1 + perlin(x*0.25, y*0.25, random=self.random))/2
        #self.terrain = np.fromfunction(g, self.world_shape)

        #radius = min(self.world_shape//3)
        
        #hills = self.random.choice(self.terrain.size, 3)

        wh, ww = self.world_shape
        vh, vw = self.view_shape
        sz = max(wh, ww)*2+1

        gaussian = gaussian_kernel(sz, sigma=sz//8)
        gx, gy = np.random.randint(0, sz-wh), np.random.randint(0, sz-ww)

        self.terrain = np.zeros(self.world_shape)
        self.terrain = gaussian[gy:gy+wh,gx:gx+ww]
        self.terrain *= 1.0/self.terrain.max()

        self.position = (np.random.randint(0, wh-vh), np.random.randint(0, ww-vw)) 

        p = self.terrain.flatten()/np.sum(self.terrain)
        t = self.random.choice(self.terrain.size, self.num_targets, replace=False, p=p)

        print(p.min(), p.max())

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
                    r += 10
                    self.targets[i] = (ty, tx, True)

        d = all(hit for _, _, hit in self.targets)

        if d:
            r += 100

        r -= 1

        return self.observe(), r, d, {}

    def render(self, mode="rgb_array", observe=False):
        if observe:
            img = self.observe()
        else:
            vh, vw = self.view_shape
            y0, x0 = self.position
            y1, x1 = y0+vh, x0+vw

            img = self.image(hidden=True)
            img[y0:y1,x0:x1] = 1.0 - img[y0:y1,x0:x1]
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
        img[:,:,0] = self.terrain

        for ty, tx, hit in self.targets:
            if hit:
                img[ty,tx] = (0,1,0)
            elif hidden:
                img[ty,tx] = (0,0,1)

        return img

