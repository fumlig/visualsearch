import gym
import enum
import numpy as np
import pyrender as pyr
import trimesh as tri


from gym_search.utils import clamp
from gym_search.shapes import Box


class VoxelEnv(gym.Env):

    metadata = {"render.modes": ["rgb_array"]}

    class Action(enum.IntEnum):
        NONE = 0
        TRIGGER = 1
        LEFT = 2
        RIGHT = 3
        DOWN = 4
        UP = 5
        BACK = 6
        FORWARD = 7

    def __init__(
        self,
    ):
        self.view = (640, 640)
        self.shape = (32, 32, 32)
        self.renderer = pyr.OffscreenRenderer(*self.view)
        self.seed()

        self.reward_range = (-np.inf, np.inf)
        self.action_space = gym.spaces.Discrete(len(self.Action))
        self.observation_space = gym.spaces.Dict(dict(
            image=gym.spaces.Box(0, 255, (*self.view, 3), dtype=np.uint8),
        ))

    def reset(self):
        self.scene = pyr.Scene(ambient_light=[0.02, 0.02, 0.02], bg_color=[1.0, 1.0, 1.0])
        self.camera = pyr.PerspectiveCamera(yfov=np.pi/3.0, aspectRatio=1.414)
        self.scene.add(self.camera, pose=np.eye(4))
        
        box = pyr.Mesh.from_trimesh(tri.primitives.Box((5, 10, 15)).copy())
        self.scene.add(box, pose=np.eye(4))

        return self.observation()


    def step(self, action):
        node = self.scene.main_camera_node
        #node.translation += 
        #node.rotation += self.get_action_rotate(action)

        if action == self.Action.BACK:
            delta = ( 0, 0,-1)
        elif action == self.Action.FORWARD:
            delta = ( 0, 0, 1)
        else:
            delta = ( 0, 0, 0)
        
        if action == self.Action.LEFT:
            direction = (0, 0, 1)
            angle = np.pi/2
        elif action == self.Action.RIGHT:
            direction = (0, 0, 1)
            angle = np.pi/2
        else:
            direction = ( 0, 0, 0)
            angle = 0

        translation_mat = tri.transformations.translation_matrix(delta)
        rotation_mat = tri.transformations.rotation_matrix(angle, direction)

        node.matrix = rotation_mat.dot(node.matrix)
        #node.matrix = translation_mat.dot(node.matrix)

        print(node.translation)

        return self.observation(), 0.0, False, {}

    def render(self, mode="rgb_array"):
        color, _depth = self.renderer.render(self.scene)
        return color

    def close(self):
        self.renderer.delete()

    def seed(self, seed=None):
        self.random = np.random.default_rng(seed)
        return [seed]

    def observation(self):
        return dict(
            image=self.render(),
        )

    def get_action_meanings(self):
        return [a.name for a in self.Action]
    
    def get_keys_to_action(self):
        return {
            (ord(" "),): self.Action.TRIGGER,
            (ord("a"),): self.Action.LEFT,
            (ord("d"),): self.Action.RIGHT,
            (ord("q"),): self.Action.DOWN,
            (ord("e"),): self.Action.UP,
            (ord("s"),): self.Action.BACK,
            (ord("w"),): self.Action.FORWARD,
        }
