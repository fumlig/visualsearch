import gym
import enum
import random
import numpy as np

from collections import defaultdict
from gym_search.utils import manhattan_dist
from skimage import draw

from typing import Dict, List, Tuple, Union
from numpy.typing import ArrayLike

class Action(enum.IntEnum):
    """
    Actions for search environments.
    
    TRIGGER: indicates that a target is visible.
    UP, RIGHT, DOWN, LEFT: move the camera in each cardinal direction.
    """
    TRIGGER = 0
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4

Observation = Dict[str, ArrayLike]

class SearchEnv(gym.Env):
    """
    Search environment base class.

    The goal of an agent in this environment is to locate a set of targets in a minimum amount of time.
    The agent perceives the environment in the form of an image.
    It also observes it position in the environment at each time step.

    When a target is visible to the agent, i.e. on the same position, the agent should indicate that it is visible.
    The episode ends when this has happened once for each target.
    """

    metadata = {"render.modes": ["rgb_array"]}

    def __init__(
        self,
        shape: Tuple[int, int],
        view: Tuple[int, int],
        wrap: Union[bool, Tuple[bool, bool]] = False,
        max_steps: int = 1000,
        num_samples: Union[int, None] = None,
        first_sample: int = 0,
        time_penalty: float = 0.01,
        explore_reward: float = 0.005,
        closer_reward: float = 0.005
    ):
        """
        Initialize search environment.

        shape: Shape of search space.
        view: Shape of image observations.
        wrap: Whether position wraps around edges of search space.
        max_steps: Maximum episode length.
        num_samples: Size of seed pool used to generate environment samples.
        first_sample: Set to larger than zero to offset the seed pool used to generate environment samples. 
        time_penalty: Constant time penalty term in reward signal.
        explore_reward: Bonus reward term for exploring search space.
        closer_reward: Bonus reward term for moving towards nearest target.
        """

        self.shape = shape
        self.view = view
        self.wrap = wrap if np.array(wrap).shape == (2,) else (wrap, wrap)

        self.max_steps = max_steps
        self.num_samples = num_samples if num_samples else np.iinfo(np.int64).max
        self.first_sample = first_sample

        self.time_penalty = time_penalty
        self.explore_reward = explore_reward
        self.closer_reward = closer_reward

        self.reward_range = (-np.inf, np.inf)
        self.action_space = gym.spaces.Discrete(len(Action))
        self.observation_space = gym.spaces.Dict(dict(image=gym.spaces.Box(0, 255, (*self.view, 3), dtype=np.uint8), position=gym.spaces.MultiDiscrete(self.shape)))


    def reset(self, seed: int = None, return_info: bool = False) -> None:
        """
        Reset search environment.

        seed: Seed used to generate new environment sample.
        return_info: Whether to return environment information.
        """
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        seed = self.np_random.integers(self.first_sample, self.num_samples)

        self.scene, self.position, self.targets = self._generate(seed)
        self.initial = self.position
        self.hits = [False for _ in range(len(self.targets))]
        self.path = [tuple(self.position)]
        self.visited = {tuple(self.position)}
        self.actions = []
        self.num_steps = 0
        self.counters = defaultdict(int)
        
        obs = self.observation()

        if return_info:
            info = {
                "position": self.position,
                "initial": self.initial,
                "targets": self.targets,
                "hits": self.hits,
                "path": self.path,
                "actions": self.actions,
                "success": all(self.hits),
                "counter": self.counters
            }

            return obs, info
        else:
            return obs

    def step(self, action: Action) -> Tuple[Observation, float, bool, dict]:
        """
        Step environment forward.

        action: The action to perform.
        """

        trigger = action == Action.TRIGGER
        last_position = tuple(self.position)
        step = self.get_action_step(action)
        self.position, invalid = self.get_next_position(step)
        revisit = tuple(self.position) in self.visited
        distances = [(manhattan_dist(last_position, target), tuple(target)) for target, hit in zip(self.targets, self.hits) if not hit]
        last_distance, nearest = min(distances)
        closer = manhattan_dist(self.position, nearest) < last_distance

        hits = 0

        if action == Action.TRIGGER:
            for i in range(len(self.targets)):
                if self.hits[i]:
                    continue

                if self._visible(self.targets[i]):
                    self.hits[i] = True
                    hits += 1

        self.num_steps += 1
        self.counters["triggers"] += trigger
        self.counters["revisits"] += revisit and not trigger
        self.counters["explored"] += not revisit
        self.counters["invalid"] += invalid
        
        self.path.append(tuple(self.position))
        self.visited.add(tuple(self.position))
        self.actions.append(action)

        obs = self.observation()
        rew = hits

        done = all(self.hits) or (self.max_steps and self.num_steps >= self.max_steps)
        info = {
            "position": self.position,
            "initial": self.initial,
            "targets": self.targets,
            "hits": self.hits,
            "path": self.path,
            "actions": self.actions,
            "success": all(self.hits),
            "counter": self.counters
        }

        if self.time_penalty:
            rew -= self.time_penalty

        if self.explore_reward and not revisit:
            rew += self.explore_reward

        if self.closer_reward and closer:
            rew += self.closer_reward

        return obs, rew, done, info


    def render(self, mode: str = "rgb_array") -> ArrayLike:
        """
        Render an image of environment.

        mode: Render mode (only "rgb_array" supported).
        """

        image = self.scene.copy()

        coords = tuple(draw.rectangle_perimeter(self._scale(self.position), extent=self.view, shape=self._scale(self.shape)))
        image[coords] = (0, 0, 0)
        
        image = image.astype(np.uint8)
        
        return image


    def observation(self) -> Observation:
        """
        Observation of current environment state.
        
        return: environment observation.
        """
        y0, x0 = np.array(self.position)*self.view
        y1, x1 = y0 + self.view[0], x0 + self.view[1]
        obs = self.scene[y0:y1,x0:x1]
        
        return dict(image=obs, position=self.position)


    def _visible(self, target: Tuple[int, int]) -> bool:
        """
        Whether point in search space is visible.

        target: Point in search space.
        return: Whether point is visible.
        """
        return np.all(self.position == target)

    def _generate(self, seed: int) -> Tuple[ArrayLike, Tuple[int, int], List[Tuple[int, int]]]:
        """
        Generate new sample of environment.

        seed: Seed to generate environment for.
        return: Environment appearance image, starting position and list of target positions.
        """
        raise NotImplementedError

    def _scale(self, position: Tuple[int, int]):
        """
        Point in search space to pixel position in environment.

        position: Position in search space to get scaled position of.
        return: Scaled position. 
        """
        return np.array(position)*self.view


    def get_action_meanings(self):
        """
        List of action names.
        """
        return [a.name for a in Action]
    
    def get_keys_to_action(self):
        """
        Mapping from keys to actions.
        """
        return {
            (ord(" "),): Action.TRIGGER,
            (ord("w"),): Action.UP,
            (ord("d"),): Action.RIGHT,
            (ord("s"),): Action.DOWN,
            (ord("a"),): Action.LEFT,
        }

    def get_action_step(self, action):
        """
        Mapping from actions to cardinal directions.
        """
        return {
            Action.UP:      (-1, 0),
            Action.RIGHT:   ( 0, 1),
            Action.DOWN:    ( 1, 0),
            Action.LEFT:    ( 0,-1)
        }.get(action, (0, 0))

    def get_next_position(self, step):
        """
        Next position of agent given action direction.
        """
        y, x = self.position + step
        h, w = self.shape

        wrap_y, wrap_x = self.wrap

        if wrap_y:
            y %= h
        else:
            y = np.clip(y, 0, h-1)
        
        if wrap_x:
            x %= w
        else:
            x = np.clip(x, 0, w-1)

        position = np.array([y, x])

        """
        if self.wrap:
            position %= self.shape
        else:
            position = np.clip(position, (0, 0), np.array(self.shape) - (1, 1))
        """

        invalid = tuple(self.position + step) != tuple(position)

        return position, invalid

    def get_random_action(self, detect=True):
        """
        Sample random action.

        detect: Whether to automatically detect targets.
        return: Action.
        """
        if detect and any([self._visible(target) and not hit for target, hit in zip(self.targets, self.hits)]):
            return 0
        
        return random.choice(range(1, self.action_space.n))

    def get_greedy_action(self, detect=True):
        """
        Select action that explores search space greedily.

        detect: Whether to automatically detect targets.
        return: Action.
        """

        if detect and any([self._visible(target) and not hit for target, hit in zip(self.targets, self.hits)]):
            return 0

        valid = []

        for action in range(1, self.action_space.n):
            step = self.get_action_step(action)
            position, _ = self.get_next_position(step)
            
            if not tuple(position) in self.visited:
                valid.append(action)

        if not valid:
            return random.choice(range(1, self.action_space.n))

        return random.choice(valid)

    def get_exhaustive_action(self, detect=True):
        """
        Select action that exhaustively covers search space.

        detect: Whether to automatically detect targets.
        return: Action.
        """

        if detect and any([self._visible(target) and not hit for target, hit in zip(self.targets, self.hits)]):
            return 0

        valid = []

        for action in range(1, self.action_space.n):
            step = self.get_action_step(action)
            position, _ = self.get_next_position(step)
            
            if not tuple(position) in self.visited:
                valid.append(action)

        if not valid:
            return random.choice(range(1, self.action_space.n))
    
        return valid[0]
    
    def get_handcrafted_action(self, detect=True):
        """
        Select action from handcrafted policy.

        detect: Whether to automatically detect targets.
        return: Action.
        """

        raise NotImplemented

    def get_position(self):
        """
        Get position of agent.
        """
        return self.position

    def set_position(self, position):
        """
        Set position of agent.
        """
        self.position = position