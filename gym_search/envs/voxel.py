import gym
import enum
import numpy as np
import trimesh as tri
import pyrender as pyr
import pymartini as martini

from gym_search.utils import clamp
from gym_search.shapes import Box
from gym_search.generators import TerrainGenerator


class VoxelEnv(gym.Env):

    metadata = {"render.modes": ["rgb_array"]}

    class Action(enum.IntEnum):
        NONE = 0
        TRIGGER = 1
        LEFT = 2
        RIGHT = 3
        DOWN = 4
        UP = 5

    def __init__(
        self,
        camera_view=(512, 512),
        camera_step=0.1,
    ):
        self.camera_view = camera_view
        self.camera_step = camera_step
        
        self.terrain_size = 256
        self.terrain_height = 50

        self.renderer = pyr.OffscreenRenderer(*self.camera_view)
        self.martini = martini.Martini(self.terrain_size+1)
        self.generator = TerrainGenerator((self.terrain_size+1, self.terrain_size+1), 0)

        self.seed()

        self.reward_range = (-np.inf, np.inf)
        self.action_space = gym.spaces.Discrete(len(self.Action))
        self.observation_space = gym.spaces.Dict(dict(
            image=gym.spaces.Box(0, 255, (*self.camera_view, 3), dtype=np.uint8),
        ))

    def reset(self):
        self.scene = pyr.Scene(ambient_light=[1.0, 1.0, 1.0], bg_color=[135, 206, 235])
        self.camera = pyr.PerspectiveCamera(yfov=np.pi/5.0, aspectRatio=self.camera_view[1]/self.camera_view[0])
        self.yaw_node = pyr.Node(matrix=tri.transformations.translation_matrix((0, 50, 0)))
        self.pitch_node = pyr.Node(matrix=np.eye(4), camera=self.camera)

        self.scene.add_node(self.yaw_node)
        self.scene.add_node(self.pitch_node, parent_node=self.yaw_node)

        terrain, colors = self.generator.terrain(self.random.integers(0, 100))
        terrain *= self.terrain_height
        tile = self.martini.create_tile(terrain.astype(np.float32))
        vertices, indices = tile.get_mesh(0.5)
        
        vertices = martini.rescale_positions(vertices, terrain)
        vertices = vertices[:,[0,2,1]]
        indices = indices.reshape(-1, 3)

        uv = np.array([(u/self.terrain_size+1, (1-v)/self.terrain_size+1) for u, _, v in vertices])
        visual = tri.visual.TextureVisuals(uv=uv, image=colors)
        mesh = pyr.Mesh.from_trimesh(tri.Trimesh(vertices=vertices, faces=indices, visual=visual))

        self.scene.add(mesh, pose=tri.transformations.translation_matrix((-self.terrain_size//2, 0, -self.terrain_size/2)))

        for t in range(10):
            x, z = self.random.integers(-self.terrain_size/2+1, self.terrain_size/2-1, size=2)
            y = terrain[z, x]
            t = tri.creation.cylinder(1, 5, transform=tri.transformations.rotation_matrix(np.pi/2, (1, 0, 0)))
            t.visual = tri.visual.color.ColorVisuals(vertex_colors=[(255, 0, 0) for _ in t.vertices])
            m = pyr.Mesh.from_trimesh(t)
            self.scene.add(m, pose=tri.transformations.translation_matrix((x, y, z)))


        return self.observation()


    def step(self, action):

        if action == self.Action.NONE:
            pass
        elif action == self.Action.TRIGGER:
            pass
        elif action == self.Action.LEFT:
            self.yaw_node.matrix = tri.transformations.rotation_matrix(self.camera_step, (0, 1, 0)).dot(self.yaw_node.matrix)
        elif action == self.Action.RIGHT:
            self.yaw_node.matrix = tri.transformations.rotation_matrix(-self.camera_step, (0, 1, 0)).dot(self.yaw_node.matrix)
        elif action == self.Action.DOWN:
            self.pitch_node.matrix = tri.transformations.rotation_matrix(-self.camera_step, (1, 0, 0)).dot(self.pitch_node.matrix)
        elif action == self.Action.UP:
            self.pitch_node.matrix = tri.transformations.rotation_matrix(self.camera_step, (1, 0, 0)).dot(self.pitch_node.matrix)

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
            (ord("s"),): self.Action.DOWN,
            (ord("w"),): self.Action.UP,
        }
