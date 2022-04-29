import gym
import enum
import random
import numpy as np

from collections import defaultdict
from gym_search.shapes import Box
from gym_search.utils import manhattan_dist
from skimage import draw


class Action(enum.IntEnum):
    TRIGGER = 0
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4


class SearchEnv(gym.Env):

    metadata = {"render.modes": ["rgb_array"]}

    def __init__(
        self,
        shape,
        view,
        wrap=False,
        train_steps=1000,
        test_steps=1000,
        train_samples=None,
        test_samples=1000,
        reward_explore=False,
        reward_closer=False
    ):
        self.shape = shape
        self.view = view
        self.wrap = wrap

        self.train_steps = train_steps
        self.test_steps = test_steps
        self.train_samples = train_samples if train_samples is not None else np.iinfo(np.int64).max - test_samples
        self.test_samples = test_samples
        self.test_seed = 0

        self.reward_explore = reward_explore
        self.reward_closer = reward_closer

        self.training = True

        self.reward_range = (-np.inf, np.inf)
        self.action_space = gym.spaces.Discrete(len(Action))
        self.observation_space = gym.spaces.Dict(dict(image=gym.spaces.Box(0, 255, (*self.view, 3), dtype=np.uint8), position=gym.spaces.MultiDiscrete(self.shape)))


    def reset(self, seed=None):
        if seed is not None:
            self.np_random, seed = gym.utils.seeding.np_random(seed)

        if self.training:
            seed = self.np_random.integers(self.test_samples, self.test_samples + self.train_samples)
        else:
            seed = self.test_seed
            self.test_seed += 1
            self.test_seed %= self.test_samples

        self.scene, self.position, self.targets = self.generate(seed)
        self.initial = self.position
        self.hits = [False for _ in range(len(self.targets))]
        self.path = [tuple(self.position)]
        self.visited = {tuple(self.position)}
        self.num_steps = 0
        self.counters = defaultdict(int)
        
        return self.observation()

    def step(self, action):

        last_position = self.position
        step = self.get_action_step(action)
        self.position, invalid = self.get_next_position(step)
        revisit = tuple(self.position) in self.visited
        nearest = self.targets[np.argmin([manhattan_dist(last_position, target) for target, hit in zip(self.targets, self.hits) if not hit])]
        closer = manhattan_dist(self.position, nearest) < manhattan_dist(last_position, nearest)

        hits = 0

        if action == Action.TRIGGER:
            for i in range(len(self.targets)):
                if self.hits[i]:
                    continue

                if self.visible(self.targets[i]):
                    self.hits[i] = True
                    hits += 1

        self.num_steps += 1
        self.counters["triggers"] += action == Action.TRIGGER
        self.counters["revisits"] += revisit
        self.counters["explored"] += not revisit
        self.counters["invalid"] += invalid
        
        self.path.append(tuple(self.position))
        self.visited.add(tuple(self.position))

        obs = self.observation()
        rew = hits - 0.01

        done = all(self.hits) or self.num_steps >= self.max_steps
        info = {
            "position": self.position,
            "initial": self.initial,
            "targets": self.targets,
            "hits": self.hits,
            "path": self.path,
            "success": all(self.hits),
            "counter": self.counters
        }

        if self.reward_explore and not revisit and not hits:
            rew += 0.005

        if self.reward_closer and closer and not hits:
            rew += 0.005

        return obs, rew, done, info


    def render(self, mode="rgb_array"):
        image = self.scene.copy()

        for i in range(len(self.targets)):
            if self.hits[i]:
                #coords = tuple(draw.rectangle(self.targets[i].position, extent=self.targets[i].shape, shape=self.scale(self.shape)))
                coords = tuple(draw.rectangle_perimeter(self.scale(self.targets[i]), extent=self.view, shape=self.scale(self.shape)))

                image[coords] = (0, 255, 0)

        #for p in self.path:
        #    coords = tuple(draw.rectangle_perimeter(self.scale(p), extent=self.view, shape=self.scale(self.shape)))
        #    image[coords] = (0.5, 0.5, 0.5)

        coords = tuple(draw.rectangle_perimeter(self.scale(self.position), extent=self.view, shape=self.scale(self.shape)))
        image[coords] = (0, 0, 0)
        
        image = image.astype(np.uint8)
        
        return image

    def plot(self, ax, overlay=True, position=None):
        _position = self.position
        if position is not None: self.position = position

        img = self.render()
        obs = self.observation()

        ax.grid(color="black", linestyle='--', linewidth=0.25)
        ax.set_yticks(range(0, img.shape[0], self.view[0]))
        ax.set_xticks(range(0, img.shape[1], self.view[1]))
        ax.set_yticklabels(range(self.shape[0]))
        ax.set_xticklabels(range(self.shape[1]))

        ax.imshow(img)

        if overlay:
            axins = ax.inset_axes((0.1, 0.1, 0.25, 0.25))
            axins.imshow(obs["image"], origin="upper")
            axins.set_yticks([0, self.view[0]-1])
            axins.set_xticks([0, self.view[1]-1])
            ax.indicate_inset([*self.scale(self.position)[::-1], *self.view[::-1]], axins, edgecolor="black")
        else:
            ax.set_yticks([])
            ax.set_xticks([])

        self.position = _position

    def observation(self):
        y0, x0 = np.array(self.position)*self.view
        y1, x1 = y0 + self.view[0], x0 + self.view[1]
        obs = self.scene[y0:y1,x0:x1]
        return dict(image=obs, position=self.position)

    def visible(self, target):
        #return Box(*self.scale(self.position), *self.view).overlap(target) > 0
        return np.all(self.position == target)

    def generate(self, seed):
        raise NotImplementedError


    def train(self, mode=True):
        self.training = mode
    
    def test(self):
        self.train(False)


    @property
    def max_steps(self):
        if self.training:
            return self.train_steps
        else:
            return self.test_steps


    def scale(self, position):
        return np.array(position)*self.view


    def get_action_meanings(self):
        return [a.name for a in Action]
    
    def get_keys_to_action(self):
        return {
            (ord(" "),): Action.TRIGGER,
            (ord("w"),): Action.UP,
            (ord("d"),): Action.RIGHT,
            (ord("s"),): Action.DOWN,
            (ord("a"),): Action.LEFT,
        }

    def get_action_step(self, action):
        return {
            Action.UP:      (-1, 0),
            Action.RIGHT:   ( 0, 1),
            Action.DOWN:    ( 1, 0),
            Action.LEFT:    ( 0,-1)
        }.get(action, (0, 0))

    def get_next_position(self, step):
        position = self.position + step

        if self.wrap:
            position %= self.shape
        else:
            position = np.clip(position, (0, 0), np.array(self.shape) - (1, 1))

        invalid = tuple(self.position + step) != tuple(position)

        return position, invalid

    def get_random_action(self, detect=False):
        if not detect and any([self.visible(target) and not hit for target, hit in zip(self.targets, self.hits)]):
            return 0
        
        return random.choice(range(1, self.action_space.n))

    def get_greedy_action(self, detect=False, deterministic=True):
        if not detect and any([self.visible(target) and not hit for target, hit in zip(self.targets, self.hits)]):
            return 0

        valid = []

        for action in range(1, self.action_space.n):
            step = self.get_action_step(action)
            position, _ = self.get_next_position(step)
            
            if not tuple(position) in self.visited:
                valid.append(action)

        if not valid:
            return random.choice(range(1, self.action_space.n))
    
        if deterministic:
            return valid[0]

        return random.choice(valid)