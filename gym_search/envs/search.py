import gym
import enum
import numpy as np

from gym.utils import seeding
from gym_search.utils import clamp
from gym_search.shapes import Rect
from gym_search.palette import add_with_alpha
from skimage import draw

class SearchEnv(gym.Env):

    metadata = {"render.modes": ["rgb_array"]}

    class Action(enum.IntEnum):
        NONE = 0
        TRIGGER = 1
        NORTH = 2
        EAST = 3
        SOUTH = 4
        WEST = 5
        # todo: done action?

    def __init__(
        self,
        generator,
        view_shape=(32, 32), 
        step_size=1, 
        random_pos=False,
        rew_exploration=True,
        max_steps=1000
    ):
        self.generator = generator
        self.shape = generator.shape
        self.view = Rect(0, 0, view_shape[0], view_shape[1])
        self.step_size = step_size
        self.random_pos = random_pos
        self.rew_exploration = rew_exploration
        self.max_steps = max_steps

        self.reward_range = (-np.inf, np.inf)
        self.action_space = gym.spaces.Discrete(len(self.Action))
        self.observation_space = gym.spaces.Dict(dict(
            time=gym.spaces.Discrete(max_steps),
            image=gym.spaces.Box(0, 1, (*self.view.shape, 3)),
            #img=gym.spaces.Box(0, 255, (*self.view.shape, 3), np.uint8),
            position=gym.spaces.Discrete(self.shape[0]*self.shape[1]),
            visited=gym.spaces.Box(0, 1, self.shape),
            #triggered=gym.spaces.Box(0, 1, self.shape),
        ))
        # this observation space makes some sense, position = pan tilt percentage (the world is only what can possibly be seen by the sensor)

        self.seed()


    def reset(self):
        h, w = self.shape
        
        if self.random_pos:
            y, x = self.random.integers(0, (h-self.view.h+1)//self.step_size)*self.step_size, self.random.integers(0, (w-self.view.w+1)//self.step_size)*self.step_size
        else:
            y, x = 0, 0

        self.view.pos = (y, x)
        self.terrain, self.targets = self.generator.generate()
        self.hits = [False for _ in range(len(self.targets))]
        self.visited = np.full(self.shape, False)
        self.triggered = np.full(self.shape, False)
        self.path = [self.view.pos]
        self.num_steps = 0

        return self._observe()


    def step(self, action):
        h, w = self.shape
        dy, dx = {
            self.Action.NORTH:   (-1, 0),
            self.Action.EAST:    ( 0, 1),
            self.Action.SOUTH:   ( 1, 0),
            self.Action.WEST:    ( 0,-1)
        }.get(action, (0, 0))

        y = clamp(self.view.y+dy*self.step_size, 0, h-self.view.h)
        x = clamp(self.view.x+dx*self.step_size, 0, w-self.view.w)

        self.view.pos = (y, x)

        rew = -2

        if action == self.Action.TRIGGER:
            rew -= 3

            for i in range(len(self.targets)):                
                if self.hits[i]:
                    continue
                
                if self.view.overlap(self.targets[i]) > 0:
                    rew += 10
                    self.hits[i] = True
        else:
            if self.rew_exploration and not self.visited[self.view.pos]:
                rew += 1

        self.visited[self.view.pos] = True
        self.triggered[self.view.pos] = True
        self.path.append(self.view.pos)

        done = all(self.hits)

        if done:
            rew = 100

        obs = self._observe()

        self.num_steps += 1

        if self.num_steps == self.max_steps:
            done = True

        return obs, rew, done, {}

    def render(self, mode="rgb_array", observe=False):
        if observe:
            img = self._observe()["img"]
        else:
            img = self._image(show_targets=True)
            rect_coords = tuple(draw.rectangle(self.view.pos, extent=self.view.shape, shape=self.shape))
            img[rect_coords] = add_with_alpha(img[rect_coords], (0, 0, 0), 0.25)
            img = img.astype(np.uint8)

        return img

    def close(self):
        pass

    def seed(self, seed=None):
        self.random = np.random.default_rng(seed)
        self.generator.seed(seed)
        return [seed]

    def _observe(self):
        y0, x0, y1, x1 = self.view.corners()
        h, w = self.shape
        img = self._image()
        obs = img[y0:y1,x0:x1]

        return dict(
            time=self.num_steps,
            image=(obs/255).astype(dtype=float),
            #img=obs,
            position=y0*w+x0,
            visited=self.visited.astype(dtype=float),
            #triggered=self.triggered
        )


    def _image(self, show_targets=False, show_hits=True, show_path=False):
        img = self.terrain.copy()

        if show_targets or show_hits:
            for i in range(len(self.targets)):
                coords = tuple(draw.rectangle(self.targets[i].pos, extent=self.targets[i].shape, shape=self.shape))

                if show_hits and self.hits[i]:
                    img[coords] = add_with_alpha(img[coords], (0, 255, 0), 0.5)
                elif show_targets:
                    img[coords] = add_with_alpha(img[coords], (255, 0, 0), 0.5)

        if show_path:
            for i, (y, x) in enumerate(self.path):
                rr, cc = draw.disk((y+self.view.h//2-1, x+self.view.w//2-1), min(self.view.shape)//8)
                img[rr, cc] = add_with_alpha(img[rr, cc], (0, 0, 0), 0.25+0.25*i/len(self.path))

        return img

    def get_action_meanings(self):
        return [a.name for a in self.Action]
    
    def get_keys_to_action(self):
        return {
            (ord(" "),): self.Action.TRIGGER,
            (ord("w"),): self.Action.NORTH,
            (ord("d"),): self.Action.EAST,
            (ord("s"),): self.Action.SOUTH,
            (ord("a"),): self.Action.WEST,
        }

