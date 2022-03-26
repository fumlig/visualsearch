from argparse import Action
import gym
import enum
import numpy as np

from gym_search.utils import clamp, euclidean_dist, manhattan_dist
from gym_search.shapes import Box
from gym_search.palette import add_with_alpha
from skimage import draw

"""
We need to be able to create held out sets
Can follow the pattern in PyTorch of setting environment in train and test mode, env.train()/env.test()
The environment creates two seed pools which it switches between.
It could also make other changes to ensure that a tester does not access information it should not have
Should we update to new gym api?
"""


class SearchEnv(gym.Env):

    metadata = {"render.modes": ["rgb_array"]}

    class Action(enum.IntEnum):
        NONE = 0 # todo: remove... if we choose deterministic, it always chooses this action for some reason
        TRIGGER = 1
        NORTH = 2
        EAST = 3
        SOUTH = 4
        WEST = 5
        # todo: done action?
        # todo: either trigger has a cost, which means that some expensive step has to be made to check if something is there,
        # or the trigger is implicit, each observation corresponds to the expensive step

    def __init__(
        self,
        generator,
        view_shape=(32, 32), 
        step_size=1, 
        random_pos=True,
        rew_exploration=True,
        max_steps=1000
    ):
        self.generator = generator
        self.shape = generator.shape
        self.view = Box(0, 0, view_shape[0], view_shape[1])
        self.step_size = step_size
        self.random_pos = random_pos
        self.rew_exploration = rew_exploration
        self.max_steps = max_steps

        self.reward_range = (-np.inf, np.inf)
        self.action_space = gym.spaces.Discrete(len(self.Action))
        self.observation_space = gym.spaces.Dict(dict(
            image=gym.spaces.Box(0, 255, (*self.view.shape, 3), dtype=np.uint8),
            
            #time=gym.spaces.Discrete(self.max_steps),
            #position=gym.spaces.Discrete(np.prod(self.scaled_shape)),
            #visible=gym.spaces.Box(0, 1, self.scaled_shape),
            #visited=gym.spaces.Box(0, 1, self.scaled_shape),
            #triggered=gym.spaces.Box(0, 1, self.scaled_shape)
        ))
        
        self.seed()

    def reset(self):
        h, w = self.shape
        
        if self.random_pos:
            y, x = self.random.integers(0, (h-self.view.h+1)//self.step_size)*self.step_size, self.random.integers(0, (w-self.view.w+1)//self.step_size)*self.step_size
        else:
            y, x = 0, 0

        self.view.pos = (y, x)
        self.terrain, self.targets = self.generator.sample()
        self.hits = [False for _ in range(len(self.targets))]
        self.path = [self.view.pos]
        self.num_steps = 0

        self.visible = np.full(self.scaled_shape, False)
        self.visited = np.full(self.scaled_shape, False)
        self.triggered = np.full(self.scaled_shape, False)

        self.visited[self.scaled_position] = True
        self.visible[self.scaled_position] = True

        self.last_dist = euclidean_dist(self.view.center(), self.targets[0].center())

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

        self.view.pos = (y, x)
        self.path.append(self.view.pos)

        revisit = self.visited[self.scaled_position]

        self.visible = np.full(self.scaled_shape, False)
        self.visited[self.scaled_position] = True
        self.visible[self.scaled_position] = True
        
        if action == self.Action.TRIGGER:
            self.triggered[self.scaled_position] = True

        dist = np.inf
        hit = False

        for i in range(len(self.targets)):
            if self.hits[i]:
                continue

            if action == self.Action.TRIGGER and self.view.overlap(self.targets[i]):
                self.hits[i] = True
                hit = True

            d = euclidean_dist(self.view.center(), self.targets[i].center())
            if d < dist:
                dist = d
        
        if hit:
            rew = 10 # previously 5, should not matter
        elif revisit:
            rew = -2.5
        else:
            rew = -1

        self.last_dist = dist
        self.num_steps += 1

        obs = self.observation()
        done = all(self.hits) or self.num_steps == self.max_steps
    
        return obs, rew, done, {}

    def render(self, mode="rgb_array", show_view=True, show_targets=True, show_hits=True, show_path=True):
        # todo: show_path gets slow when the path is long

        img = self.terrain.copy()

        if show_targets or show_hits:
            for i in range(len(self.targets)):
                coords = tuple(draw.rectangle(self.targets[i].pos, extent=self.targets[i].shape, shape=self.shape))

                if show_hits and self.hits[i]:
                    img[coords] = add_with_alpha(img[coords], (0, 255, 0), 0.5)
                elif show_targets:
                    img[coords] = add_with_alpha(img[coords], (255, 0, 0), 0.5)

        if show_path:
            for i, pos in enumerate(self.path):
                coords = tuple(draw.rectangle_perimeter(pos, extent=self.view.extent, shape=self.shape))
                img[coords] = add_with_alpha(img[coords], (127, 127, 127), 0.25+0.5*i/len(self.path))

        if show_view:
            coords = tuple(draw.rectangle_perimeter(self.view.pos, extent=self.view.extent, shape=self.shape))
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

        return dict(
            image=obs,
            
            #time=self.num_steps,
            #position=self.scaled_position[0]*self.scaled_shape[1] + self.#scaled_position[1],
            #visible=self.visible,
            #visited=self.visited,
            #triggered=self.triggered,
        )

    @property
    def scaled_shape(self):
        h = self.shape[0] // self.view.shape[0]
        w = self.shape[1] // self.view.shape[1]
        return h, w

    @property
    def scaled_position(self):
        y = self.view.pos[0] // self.view.shape[0]
        x = self.view.pos[1] // self.view.shape[1]
        return y, x


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
