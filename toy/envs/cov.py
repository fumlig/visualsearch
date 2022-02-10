import gym
import numpy as np
import cv2 as cv

from gym.utils import seeding
from utils import perlin_noise_2d

class CovEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    ACTIONS = [
        ( 0, 0),
        (-1, 0),
        ( 0,-1),
        ( 1, 0),
        ( 0, 1),
    ]

    KEYS = [
        ord(" "),
        ord("w"),
        ord("a"),
        ord("s"),
        ord("d"),
    ]

    def __init__(self, width=50, height=50, radius=20, seed=0):
        self.shape = (height, width)
        self.radius = radius
        self.player = None
        self.tiles = None

        self.seed(seed)

        noise = perlin_noise_2d(self.shape, (height//5, width//5), random=self.random)
        self.map = np.clip(noise - 0.5, -1.0, 1.0)

        self.reward_range = (-1.0, 1.0)
        self.action_space = gym.spaces.Discrete(len(self.ACTIONS))
        self.observation_space = gym.spaces.Box(0, 255, (radius*2+1, radius*2+1, 3), dtype=np.uint8)

    def step(self, action):
        a = self.ACTIONS[action]
        h, w = self.shape
        y, x = self.player

        clamp = lambda x, lo, hi: max(min(x, hi), lo)

        y = clamp(y+a[0], 0, h-1)
        x = clamp(x+a[1], 0, w-1)

        self.player = (y, x)

        tile = self.tiles[self.player]
        r = 1 if tile > 0 else -1
        self.tiles[self.player] = 0

        done = np.sum(self.tiles > 0) == 0

        return self.observe(), r, done, {"shape": self.shape, "player": self.player}

    def reset(self):
        h, w = self.shape
        
        self.player = (h//2, w//2)
        #self.tiles = self.random.uniform(0, 1, size=self.shape)
        #self.tiles = np.ones(self.shape)

        #noise = perlin_noise_2d(self.shape, (h//5, w//5), random=self.random)
        #self.tiles = np.clip(noise - 0.5, -1.0, 1.0)

        self.tiles = self.map.copy()

        return self.observe()


    def render(self, mode="human"):
        y, x = self.player
        h, w = self.shape
        r = self.radius
        dark = 0.5

        img = self.view()
        img *= dark
        img[y:y+r*2+1,x:x+r*2+1] *= 1/dark

        img = img.astype(dtype=np.uint8)

        if mode == "rgb_array":
            return img
        else:
            fix = img.copy()
            fix[:,:,0] = img[:,:,2]
            fix[:,:,2] = img[:,:,0]
            fix = cv.resize(fix, (h*2, w*2), interpolation=cv.INTER_NEAREST)
            cv.imshow("coverage", fix)
            cv.waitKey(1)

    def seed(self, seed):
        self.random, _ = seeding.np_random(seed)
        return [seed]


    def close(self):
        pass

    def observe(self):
        r = self.radius
        y, x = self.player

        view = self.view()
        obs = view[y:y+r*2+1,x:x+r*2+1]

        return obs

    def view(self):
        h, w = self.shape
        r = self.radius
        y, x = self.player

        pad = np.zeros((h+2*r, w+2*r))
        pad[r:-r, r:-r] = self.tiles

        neg = pad < 0.0
        pos = pad >= 0.0

        view = np.zeros((*pad.shape, 3))
        view[neg,0] = -pad[neg]*255 # negative reward red
        view[pos,1] = pad[pos]*255  # positive reward green
        view[y+r,x+r,2] = 255       # player blue

        return view


    def get_keys_to_action(self):
        return {(key, ): a for a, key in enumerate(self.KEYS)}
