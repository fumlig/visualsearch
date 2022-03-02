import numpy as np
import gym
from gym import Env, spaces


class PTZEnv(Env):
    """Pan-tilt-zoom camera environment."""

    metadata = {"render.modes": ["human", "ansi", "rgb_array"]}

    ACTIONS = [
        ( 0, 0, 0),
        ( 1, 0, 0),
        (-1, 0, 0),
        ( 0, 1, 0),
        ( 0,-1, 0),
        ( 0, 0, 1),
        ( 0, 0,-1),
    ]

    KEYS = [
        ord(" "),
        ord("d"),
        ord("a"),
        ord("s"),
        ord("w"),
        ord("e"),
        ord("q"),
    ]

    def __init__(self, width=10, height=10):
        self.shape = (height, width)
        self.camera = (0, 0, 0)
        self.target = (0, 0, 0)

        self.reward_range = (0.0, 1.0)
        self.action_space = spaces.Discrete(len(self.ACTIONS))
        self.observation_space = spaces.Box(0, 3, (height, width), dtype=int)

    def step(self, action):
        a = self.ACTIONS[action]

        h, w = self.shape
        x, y, z = self.camera

        clamp = lambda x, lo, hi: max(min(x, hi), lo)
        edge = x == z or x == w-z-1 or y == z or y == h-z-1

        x = clamp(x+a[0], z, w-z-1)
        y = clamp(y+a[1], z, h-z-1)
        z = max(z if edge and a[2] > 0 else z+a[2], 0)

        self.camera = (x, y, z)

        o = self.observe()
        r = self.reward()
        d = self.terminal()

        self.last_action = action
        self.last_overlap = self.overlap()

        return o, r, d, {"shape": self.shape, "camera": self.camera, "target": self.target}

    def reset(self):
        self.last_overlap = self.overlap()
        self.last_action = None

        h, w = self.shape
        self.camera = (w//2, h//2, 1)

        z = np.random.randint(1, min(self.shape)//4)
        x = np.random.randint(z, w-z)
        y = np.random.randint(z, h-z)
        self.target = (x, y, z)

        return self.observe()


    def render(self, mode="human"):
        obs = self.observe()

        if mode == "human":
            print("\n".join([" ".join(map(str, row)) for row in obs.tolist()]))
        elif mode == "ansi":
            return "\n".join([" ".join(map(str, row)) for row in obs.tolist()])
        elif mode == "rgb_array":
            h, w = self.shape

            img = np.empty((w, h, 3))
            img[obs == 0] = (0.0, 0.0, 0.0)
            img[obs == 1] = (1.0, 0.0, 0.0)
            img[obs == 2] = (0.0, 1.0, 0.0)

            return img

    def reward(self):
        #return 1 if self.overlap() == 1 else 0
        return self.overlap() - 1.0

    def observe(self):
        visible = self._visible_mask()
        target = self._target_mask()

        obs = np.zeros(self.shape, dtype=int)
        obs += visible                            # ones where camera is looking to indicate position
        obs += np.logical_and(visible, target)    # twos where camera is looking at target

        return obs

    def terminal(self):
        return self.camera == self.target

    def overlap(self):
        visible = self._visible_mask()
        target = self._target_mask()
        # intersection over union
        intersection = np.logical_and(visible, target)
        union = np.logical_or(visible, target)
        return np.sum(intersection) / np.sum(union)

    def get_keys_to_action(self):
        return {(key, ): a for a, key in enumerate(self.KEYS)}

    def _visible_mask(self):
        x, y, z = self.camera
        mask = np.full(self.shape, 0, dtype=int)
        mask[y-z:y+z+1,x-z:x+z+1] = 1
        return mask

    def _target_mask(self):
        x, y, z = self.target
        mask = np.full(self.shape, 0, dtype=int)
        mask[y-z:y+z+1,x-z:x+z+1] = 1
        return mask
