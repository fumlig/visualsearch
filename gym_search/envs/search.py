import gym
import enum
import numpy as np

from gym.utils import seeding
from gym_search.utils import clamp
from gym_search.terrain import uniform_terrain


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
        num_targets=3,
        terrain_func=uniform_terrain,
        rew_exploration=True,
        max_steps=1000,
        seed=0
    ):
        self.seed(seed)
        self.world_shape = world_shape
        self.view_shape = view_shape
        self.step_size = step_size
        self.num_targets = num_targets
        self.terrain_func = terrain_func
        self.rew_exploration = rew_exploration
        self.max_steps = max_steps

        self.reward_range = (-np.inf, np.inf)
        self.action_space = gym.spaces.Discrete(len(self.Action))
        self.observation_space = gym.spaces.Dict(dict(
            #t=gym.spaces.Discrete(max_steps),
            img=gym.spaces.Box(0, 1, (*self.view_shape, 3)),
            pos=gym.spaces.Discrete(self.world_shape[0]*self.world_shape[1])
            #pos=gym.spaces.Box(0, 1, (2,)),
        ))
        # this observation space makes some sense, position = pan tilt percentage (the world is only what can possibly be seen by the sensor)


    def reset(self):
        wh, ww = self.world_shape
        vh, vw = self.view_shape
        
        self.terrain = self.terrain_func(self.world_shape, self.random)
        self.position = (self.random.randint(0, (wh-vh+1)//self.step_size)*self.step_size, self.random.randint(0, (ww-vw+1)//self.step_size)*self.step_size) 
        self.visited = np.full(self.world_shape, False)

        p = self.terrain.flatten()/np.sum(self.terrain)
        t = self.random.choice(self.terrain.size, self.num_targets, replace=False, p=p)

        self.targets = [(i//ww, i%ww, False) for i in t]

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

            img = np.ones((*self.world_shape, 3))#self.image(hidden=True)*0.5
            img[y0:y1,x0:x1] = self.image(hidden=True)[y0:y1,x0:x1]

        img = img*255
        img = img.astype(dtype=np.uint8)
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

        img = self.image(hidden=True)
        obs = img[y0:y1,x0:x1,:]

        return dict(
            #t=self.num_steps,
            img=obs,
            pos=y0*ww+x0
            #pos=np.array([y0/wh, x0/ww]),
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

class PrettySearchEnv(SearchEnv):    

    GRASS   = (0.0, 1.0, 0.0)
    WATER   = (0.0, 0.0, 1.0)
    SAND    = (0.0, 1.0, 1.0)
    ROCK    = (0.5, 0.5, 0.5)

    def __init__(self, *args, **kwargs):
        super(PrettySearchEnv, self).__init__(*args, **kwargs)
    
    def image(self, hidden=False):
        img = np.zeros((*self.world_shape, 3))
        img[self.terrain <= 1.0] = self.ROCK
        img[self.terrain <= 0.75] = self.GRASS
        img[self.terrain <= 0.50] = self.SAND
        img[self.terrain <= 0.25] = self.WATER
        img[:,:,0] *= self.terrain
        img[:,:,1] *= self.terrain
        img[:,:,2] *= self.terrain

        for ty, tx, hit in self.targets:
            if hit:
                img[ty,tx] = (1,1,0)
            elif hidden:
                img[ty,tx] = (1,0,0)

        return img