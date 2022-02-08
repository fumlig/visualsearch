import numpy as np
import gym

from gym.utils import seeding

class CovEnv(gym.Env):
    metadata = {"render.modes": ["rgb_array"]}

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

    def __init__(self, width=10, height=10, seed=1337):
        self.shape = (height, width)
        self.player = (0, 0)
        self.tiles = None

        self.seed(seed)

        self.reward_range = (0.0, 1.0)
        self.action_space = gym.spaces.Discrete(len(self.ACTIONS))
        self.observation_space = gym.spaces.Box(0, 1, (2, *self.shape), dtype=np.uint8)

    def step(self, action):
        a = self.ACTIONS[action]
        h, w = self.shape
        r, c = self.player

        clamp = lambda x, lo, hi: max(min(x, hi), lo)

        r = clamp(r+a[0], 0, h-1)
        c = clamp(c+a[1], 0, w-1)

        self.player = (r, c)

        obs = self.observe()
        rew = self.tiles[self.player] - 1
        done = np.sum(self.tiles) == 0

        self.tiles[self.player] = 0

        return obs, rew, done, {"shape": self.shape, "player": self.player}

    def reset(self):
        h, w = self.shape
        
        self.player = (h//2, w//2)
        #self.tiles = self.random.uniform(0, 1, size=self.shape)
        self.tiles = np.ones(self.shape)

        return self.observe()


    def render(self, mode="rgb_array"):
        h, w = self.shape
        r, c = self.player

        img = np.empty((h, w, 3))
        img[:,:,0] = img[:,:,1] = img[:,:,2] = self.tiles
        img[r,c,:] = (1.0, 0.0, 0.0)
        
        return img

    def seed(self, seed):
        self.random, _ = seeding.np_random(seed)
        return [seed]

    def observe(self):
        obs = np.zeros((2, *self.shape))
        obs[0] = self.tiles
        obs[1] = 0
        obs[1,self.player[0],self.player[1]] = 1
        return obs

    def get_keys_to_action(self):
        return {(key, ): a for a, key in enumerate(self.KEYS)}
