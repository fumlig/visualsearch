import os
import gym
import enum
import numpy as np
import trimesh as tri
import pyrender as pyr
import pymartini as martini

from gym_search.utils import clamp
from gym_search.shapes import Box
from gym_search.generators import TerrainGenerator
from gym_search.palette import BLUE_MARBLE


os.environ["PYOPENGL_PLATFORM"] = "egl"


class CameraEnv(gym.Env):

    metadata = {"render.modes": ["rgb_array"]}

    class Action(enum.IntEnum):
        NONE = 0
        TRIGGER = 1
        LEFT = 2
        RIGHT = 3
        DOWN = 4
        UP = 5
        IN = 6
        OUT = 7

    def __init__(
        self,
        view_size=(64, 64),
        view_steps=64,
        terrain_size=256,
        terrain_height=16,
        num_targets=10,
        num_distractors=100,
    ):
        self.view_size = view_size
        self.view_steps = view_steps
        self.terrain_size = terrain_size
        self.terrain_height = terrain_height
        self.num_targets = num_targets

        self.camera_zoom_in = 2*np.pi/view_steps
        self.camera_zoom_out = self.camera_zoom_in*4
        self.camera_yaw = 0
        self.camera_pitch = 0
        self.camera_step = self.camera_zoom_in

        self.max_steps = 1000

        self.renderer = pyr.OffscreenRenderer(*self.view_size)
        self.martini = martini.Martini(self.terrain_size+1)
        self.generator = TerrainGenerator((self.terrain_size+1, self.terrain_size+1), num_targets, num_distractors)

        self.seed()

        self.reward_range = (-np.inf, np.inf)
        self.action_space = gym.spaces.Discrete(len(self.Action))
        self.observation_space = gym.spaces.Dict(dict(
            image=gym.spaces.Box(0, 255, (*self.view_size, 3), dtype=np.uint8),
            position=gym.spaces.MultiDiscrete((self.view_steps, self.view_steps))
        ))

    def reset(self):
        self.num_steps = 0
        self.scene = pyr.Scene(ambient_light=[1.0, 1.0, 1.0], bg_color=[135, 206, 235])

        # terrain
        terrain = self.generator.terrain()
        image = self.generator.image(terrain, palette=BLUE_MARBLE)
        targets = self.generator.targets(terrain)

        self.terrain = terrain*self.terrain_height
        tile = self.martini.create_tile(self.terrain.astype(np.float32))
        vertices, indices = tile.get_mesh(0.5)
        vertices = martini.rescale_positions(vertices, self.terrain)
        vertices = vertices[:,[0,2,1]]
        indices = indices.reshape(-1, 3)

        uv = np.array([(x/self.terrain_size+1, (1-z)/self.terrain_size+1) for x, _, z in vertices])
        visual = tri.visual.TextureVisuals(uv=uv, image=image)
        mesh = pyr.Mesh.from_trimesh(tri.Trimesh(vertices=vertices, faces=indices, visual=visual))

        self.scene.add(mesh, pose=np.eye(4))

        # targets
        self.targets = []
        self.hits = []
        for z, x in targets:
            side = 1
            y = self.height(x, z)+side
            t = tri.creation.box((side, side, side))
            t.visual = tri.visual.color.ColorVisuals(vertex_colors=[(255, 0, 0) for _ in t.vertices])
            m = pyr.Mesh.from_trimesh(t)
            self.scene.add(m, pose=tri.transformations.translation_matrix((x, y, z)))

            self.targets.append((x, y, z))
            self.hits.append(False)

        # camera
        
        x = self.terrain_size//2
        z = self.terrain_size//2
        y = self.terrain_height # self.height(x, z)

        self.camera = pyr.PerspectiveCamera(yfov=self.camera_zoom_out, aspectRatio=self.view_size[1]/self.view_size[0])
        self.yaw_node = pyr.Node(matrix=tri.transformations.translation_matrix((x, y, z)))
        self.pitch_node = pyr.Node(matrix=np.eye(4), camera=self.camera)

        # debug
        #self.yaw_node.matrix = tri.transformations.rotation_matrix(-np.pi/2, (1, 0, 0), point=self.yaw_node.translation) @ self.yaw_node.matrix
        #self.yaw_node.matrix = tri.transformations.translation_matrix((0, 256, 0)) @ self.yaw_node.matrix

        self.scene.add_node(self.yaw_node)
        self.scene.add_node(self.pitch_node, parent_node=self.yaw_node)

        return self.observation()


    def step(self, action):
        if action == self.Action.LEFT:
            self.yaw_node.matrix = tri.transformations.rotation_matrix(self.camera_step, (0, 1, 0), point=self.yaw_node.translation) @ self.yaw_node.matrix
            self.camera_yaw -= 1
            self.camera_yaw %= self.view_steps
        elif action == self.Action.RIGHT:
            self.yaw_node.matrix = tri.transformations.rotation_matrix(-self.camera_step, (0, 1, 0), point=self.yaw_node.translation) @ self.yaw_node.matrix
            self.camera_yaw += 1
            self.camera_yaw %= self.view_steps
        elif action == self.Action.DOWN:
            self.pitch_node.matrix = tri.transformations.rotation_matrix(-self.camera_step, (1, 0, 0)) @ self.pitch_node.matrix
            self.camera_pitch -= 1
            self.camera_pitch %= self.view_steps
        elif action == self.Action.UP:
            self.pitch_node.matrix = tri.transformations.rotation_matrix(self.camera_step, (1, 0, 0)) @ self.pitch_node.matrix
            self.camera_pitch += 1
            self.camera_pitch %= self.view_steps
        elif action == self.Action.IN:
            self.camera.yfov = self.camera_zoom_in
        elif action == self.Action.OUT:
            self.camera.yfov = self.camera_zoom_out

        hits = 0

        if action == self.Action.TRIGGER:
            for i in range(len(self.targets)):
                if self.hits[i]:
                    continue
                    
                x, y, z = self.targets[i]

                if self.is_visible(x, y, z):
                    self.hits[i] = True
                    hits += 1

        self.num_steps += 1

        obs = self.observation()

        if hits:
            rew = hits*10
        else:
            rew = -1

        done = all(self.hits) or self.num_steps == self.max_steps


        return obs, rew, done, {}

    def render(self, mode="rgb_array"):
        flags = pyr.RenderFlags.NONE
        color, _depth = self.renderer.render(self.scene, flags)
        return color

    def close(self):
        self.renderer.delete()

    def seed(self, seed=None):
        self.random = np.random.default_rng(seed)
        return [seed]

    def observation(self):
        return dict(
            image=self.render(),
            position=(self.camera_yaw, self.camera_pitch)
        )

    def height(self, x, z):
        return self.terrain[round(x), round(z)]

    def is_visible(self, x, y, z):
        camera_node = self.scene.main_camera_node
        camera_pose = self.scene.get_pose(camera_node)
        
        proj = camera_node.camera.get_projection_matrix(*self.view_size)
        view = np.linalg.inv(camera_pose)
        position = proj @ view @ np.array([x, y, z, 1.0])
        p = position[:3] / position[3]

        print(p)

        return np.all((p >= -1.0) & (p <= 1.0))

    def get_action_meanings(self):
        return [a.name for a in self.Action]
    
    def get_keys_to_action(self):
        return {
            (ord(" "),): self.Action.TRIGGER,
            (ord("a"),): self.Action.LEFT,
            (ord("d"),): self.Action.RIGHT,
            (ord("s"),): self.Action.DOWN,
            (ord("w"),): self.Action.UP,
            (ord("q"),): self.Action.OUT,
            (ord("e"),): self.Action.IN,
        }
