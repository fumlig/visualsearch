import gym
import enum
import numpy as np

from gym_search.utils import clamp, euclidean_dist, manhattan_dist, travel_dist
from gym_search.shapes import Box
from gym_search.palette import add_with_alpha
from gym_search.envs.search import SearchEnv
from skimage import draw
from collections import defaultdict


class GaussianEnv(SearchEnv):
    
    def __init__(
        self,
        shape,
        view,
        wrap=False,
        train_steps=1000,
        test_steps=5000,
        train_samples=None,
        test_samples=1000
    ):
        super().__init__(shape, view, wrap, train_steps, test_steps, train_samples, test_samples)
    
    def render(self, mode="rgb_array"):
        raise NotImplementedError

    def observation(self):
        raise NotImplementedError

    def visible(self, target):
        raise NotImplementedError

    def generate(self, seed):
        raise NotImplementedError


class SearchEnv(gym.Env):

    metadata = {"render.modes": ["rgb_array"]}

    class Action(enum.IntEnum):
        NONE = 0
        TRIGGER = 1
        NORTH = 2
        EAST = 3
        SOUTH = 4
        WEST = 5

    def __init__(self, generator, view_shape=(32, 32), step_size=1):
        self.generator = generator
        self.shape = generator.shape
        self.view = Box(0, 0, view_shape[0], view_shape[1])
        self.step_size = step_size

        self.train_steps = 1000
        self.test_steps = 5000

        self.reward_range = (-np.inf, np.inf)
        self.action_space = gym.spaces.Discrete(len(self.Action))
        self.observation_space = gym.spaces.Dict(dict(image=gym.spaces.Box(0, 255, (*self.view.shape, 3), dtype=np.uint8), position=gym.spaces.MultiDiscrete(self.scaled_shape)))
        
        self.train()
        self.seed()

    def reset(self):
        h, w = self.shape
        y = self.random.integers(0, (h-self.view.h+1)//self.step_size)*self.step_size
        x = self.random.integers(0, (w-self.view.w+1)//self.step_size)*self.step_size

        self.view.position = (y, x)
        self.terrain, self.targets = self.generator.sample()
        self.hits = [False for _ in range(len(self.targets))]
        self.path = [self.view.position]
        self.num_steps = 0

        self.visible = np.full(self.scaled_shape, False)
        self.visited = np.full(self.scaled_shape, False)
        self.triggered = np.full(self.scaled_shape, False)

        self.visited[self.scaled_position] = True
        self.visible[self.scaled_position] = True

        self.last_dist = euclidean_dist(self.view.center(), self.targets[0].center())
        self.optimal_steps = 0 # int(travel_dist(map(self.normalize_position, [self.view.position] + [target.position for target in self.targets]))) + len(self.targets)

        self.counters = defaultdict(int)

        return self.observation()


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

        self.view.position = (y, x)
        self.path.append(self.view.position)

        revisit = self.visited[self.scaled_position]
        retrigger = self.triggered[self.scaled_position] and action == self.Action.TRIGGER

        self.visible = np.full(self.scaled_shape, False)
        self.visited[self.scaled_position] = True
        self.visible[self.scaled_position] = True
        
        if action == self.Action.TRIGGER:
            self.triggered[self.scaled_position] = True

        dist = np.inf
        hits = 0

        for i in range(len(self.targets)):
            if self.hits[i]:
                continue

            if action == self.Action.TRIGGER and self.view.overlap(self.targets[i]):
                self.hits[i] = True
                hits += 1

            d = euclidean_dist(self.view.center(), self.targets[i].center())
            if d < dist:
                dist = d
        
        if hits:
            rew = hits*10
        else:
            rew = -1

        self.last_dist = dist
        self.num_steps += 1

        self.counters["revisits"] += revisit
        self.counters["retriggers"] += retrigger
        self.counters["redundant_steps"] += self.num_steps > self.optimal_steps

        obs = self.observation()
        done = all(self.hits) or self.num_steps == self.max_steps
        info = dict(
            counters=self.counters,
            path=self.path,
            success=all(self.hits),
            shortest=self.optimal_steps
        )


        return obs, rew, done, info

    def render(self, mode="rgb_array", show_view=True, show_targets=True, show_hits=True, show_path=True):
        # todo: show_path gets slow when the path is long

        img = self.terrain.copy()

        if show_targets or show_hits:
            for i in range(len(self.targets)):
                coords = tuple(draw.rectangle(self.targets[i].position, extent=self.targets[i].shape, shape=self.shape))

                if show_hits and self.hits[i]:
                    img[coords] = add_with_alpha(img[coords], (0, 255, 0), 0.5)
                elif show_targets:
                    img[coords] = add_with_alpha(img[coords], (255, 0, 0), 0.5)

        if show_path:
            for i, pos in enumerate(self.path):
                coords = tuple(draw.rectangle_perimeter(pos, extent=self.view.extent, shape=self.shape))
                img[coords] = add_with_alpha(img[coords], (127, 127, 127), 0.25+0.5*i/len(self.path))

        if show_view:
            coords = tuple(draw.rectangle_perimeter(self.view.position, extent=self.view.extent, shape=self.shape))
            img[coords] = (255, 255, 255)
            img = img.astype(np.uint8)

        return img

    def close(self):
        pass

    def seed(self, seed=None):
        self.random = np.random.default_rng(seed)
        self.generator.seed(seed)
        return [seed]

    def observation(self):
        y0, x0, y1, x1 = self.view.corners()
        obs = self.terrain[y0:y1,x0:x1]

        return dict(image=obs, position=self.scaled_position)

    @property
    def scaled_shape(self):
        h = self.shape[0] // self.view.shape[0]
        w = self.shape[1] // self.view.shape[1]
        return h, w

    @property
    def scaled_position(self):
        y = self.view.position[0] // self.view.shape[0]
        x = self.view.position[1] // self.view.shape[1]
        return y, x

    def normalize_position(self, position):
        return position[0] // self.view.shape[0], position[1] // self.view.shape[1]

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

    def train(self):
        self.max_steps = self.train_steps

    def test(self):
        self.max_steps = self.test_steps