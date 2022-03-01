import gym
import enum
import numpy as np

from gym.utils import seeding
from gym_search.utils import clamp
from gym_search.terrain import basic_terrain


class SearchEnv(gym.Env):

    metadata = {"render.modes": ["rgb_array"]}

    class Action(enum.IntEnum):
        NONE = 0
        TRIGGER = 1
        NORTH = 2
        EAST = 3
        SOUTH = 4
        WEST = 5

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
        self.world_shape = world_shape
        self.view_shape = view_shape
        self.step_size = step_size
        self.terrain_func = terrain_func
        self.rew_exploration = rew_exploration
        self.max_steps = max_steps

        self.reward_range = (-np.inf, np.inf)
        self.action_space = gym.spaces.Discrete(len(self.Action))
        self.observation_space = gym.spaces.Dict(dict(
            #t=gym.spaces.Discrete(max_steps),
            img=gym.spaces.Box(0, 255, (*self.view_shape, 3), np.uint8),
            pos=gym.spaces.Discrete(self.world_shape[0]*self.world_shape[1])
        ))
        # this observation space makes some sense, position = pan tilt percentage (the world is only what can possibly be seen by the sensor)


    def reset(self):
        wh, ww = self.world_shape
        vh, vw = self.view_shape
    
        self.terrain, targets = self.terrain_func(self.world_shape, self.random)
        self.targets = [(y, x, False) for y, x in targets]
        self.position = (self.random.randint(0, (wh-vh+1)//self.step_size)*self.step_size, self.random.randint(0, (ww-vw+1)//self.step_size)*self.step_size) 
        self.visited = np.full(self.world_shape, False)
        self.num_steps = 0

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

        rew = -1

        if self.rew_exploration and not self.visited[self.position]:
            rew += 1
        
        self.visited[self.position] = True

        if action == self.Action.TRIGGER:
            rew -= 5

            for i, t in enumerate(self.targets):
                ty, tx, hit = t

                if hit:
                    continue
                
                if px <= tx and tx < px + vw and py <= ty and ty < py + vh:
                    rew += 10
                    self.targets[i] = (ty, tx, True)

        done = all(hit for _, _, hit in self.targets)

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
            vh, vw = self.view_shape
            y0, x0 = self.position
            y1, x1 = y0+vh, x0+vw
            img = self.terrain*0.5
            img[y0:y1,x0:x1] = self.terrain[y0:y1,x0:x1]
            img = img.astype(np.uint8)

            for y, x, hit in self.targets:
                if hit:
                    img[y, x] = np.array((255, 255, 255)) - img[y, x]

        return img

    def close(self):
        pass

    def seed(self, seed=None):
        self.random, _ = seeding.np_random(seed)
        return [seed]

    def observe(self):
        wh, ww = self.world_shape
        vh, vw = self.view_shape
        y0, x0 = self.position
        y1, x1 = y0+vh, x0+vw

        img = self.terrain
        obs = img[y0:y1,x0:x1,:]

        return dict(
            #t=self.num_steps,
            img=obs,
            pos=y0*ww+x0
        )
    
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
