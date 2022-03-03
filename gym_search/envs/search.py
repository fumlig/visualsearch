import gym
import enum
import numpy as np

from gym.utils import seeding
from gym_search.utils import clamp
from gym_search.terrain import basic_terrain
from gym_search.shape import Rect


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
        world_shape=(256, 256), 
        view_shape=(32, 32), 
        step_size=32, 
        terrain_func=basic_terrain,
        rew_exploration=True,
        max_steps=1000,
        seed=0
    ):
        self.seed(seed)
        self.shape = world_shape
        self.view = Rect(0, 0, view_shape[0], view_shape[1])
        self.step_size = step_size
        self.terrain_func = terrain_func
        self.rew_exploration = rew_exploration
        self.max_steps = max_steps

        self.reward_range = (-np.inf, np.inf)
        self.action_space = gym.spaces.Discrete(len(self.Action))
        self.observation_space = gym.spaces.Dict(dict(
            #t=gym.spaces.Discrete(max_steps),
            img=gym.spaces.Box(0, 255, (*self.view.shape, 3), np.uint8),
            pos=gym.spaces.Discrete(self.shape[0]*self.shape[1])
        ))
        # this observation space makes some sense, position = pan tilt percentage (the world is only what can possibly be seen by the sensor)


    def reset(self):
        h, w = self.shape
        y, x = self.random.randint(0, (h-self.view.h+1)//self.step_size)*self.step_size, self.random.randint(0, (w-self.view.w+1)//self.step_size)*self.step_size

        self.view.pos = (y, x)
        self.terrain, self.targets = self.terrain_func(self.shape, self.random)
        self.hits = [False for _ in range(len(self.targets))]
        self.visited = np.full(self.shape, False)
        self.num_steps = 0

        return self.observe()


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

        
        self.visited[self.view.pos] = True

        rew = 0

        if action == self.Action.TRIGGER:
            rew -= 5

            for i in range(len(self.targets)):                
                if self.hits[i]:
                    continue
                
                if self.view.overlap(self.targets[i]) > 0:
                    rew += 10
                    self.hits[i] = True
        else:
            rew -= 1

            if self.rew_exploration and not self.visited[self.view.pos]:
                rew += 1

        done = all(self.hits)

        if done:
            rew += 100

        obs = self.observe()

        self.num_steps += 1

        if self.num_steps == self.max_steps:
            done = True

        return obs, rew, done, {}

    def render(self, mode="rgb_array", observe=False):
        if observe:
            img = self.observe()["img"]
        else:
            y0, x0, y1, x1 = self.view.corners()
            img = self.image()*0.5
            img[y0:y1,x0:x1] = self.image()[y0:y1,x0:x1]
            img = img.astype(np.uint8)

        return img

    def close(self):
        pass

    def seed(self, seed=None):
        self.random, _ = seeding.np_random(seed)
        return [seed]

    def observe(self):
        y0, x0, y1, x1 = self.view.corners()
        h, w = self.shape
        img = self.image()
        obs = img[y0:y1,x0:x1,:]

        return dict(
            #t=self.num_steps,
            img=obs,
            pos=y0*w+x0
        )

    def image(self, indicate_hit=True):
        img = self.terrain.copy()

        for i in range(len(self.targets)):
            if self.hits[i]:
                ty0, tx0, ty1, tx1 = self.targets[i].corners()
                img[ty0:ty1, tx0:tx1] = np.array((255, 255, 255)) - img[ty0:ty1, tx0:tx1]
        
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

