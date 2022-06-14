import numpy as np
import moderngl as gl
import pyrr as pr
import simple_3dviz as viz
import pymartini as martini
import time

from gym_search.utils import sample_coords, fractal_noise_2d, normalize
from gym_search.palette import BLUE_MARBLE, pick_color
from gym_search.envs.search import SearchEnv

from typing import Tuple

class CameraEnv(SearchEnv):
    """
    Search environment with a perspective projection camera.
    """

    def __init__(
        self,
        shape: Tuple[int] = (10, 20),
        view: Tuple[int] = (64, 64),
        terrain_size: int = 1024,
        terrain_height: int = 32,
        num_targets: int = 3,
        **kwargs,
    ):
        """
        shape: Shape of search space.
        view: Shape of image observations.
        terrain_size: Width and depth of terrain.
        terrain_height: Height of terrain.
        num_targets: Number of targets.
        **kwargs: Passed to super constructor.
        """

        super().__init__(shape, view, (False, True), **kwargs)

        self.terrain_size = terrain_size
        self.terrain_height = terrain_height
        self.num_targets = num_targets

        self.scene = viz.Scene(background=(0.75, 0.75, 1.0, 1.0), size=self.view)
        self.martini = martini.Martini(self.terrain_size+1)

    def _generate(self, seed: int):

        scene = self.scene
        
        scene.clear()
        random = np.random.default_rng(seed)

        # terrain
        exp = random.uniform(0.5, 3)

        t = time.process_time()
        noise = fractal_noise_2d((self.terrain_size, self.terrain_size), periods=(4, 4), octaves=4, seed=seed)
        #print("noise:", time.process_time() - t)
        noise = np.pad(noise, ((0, 1), (0, 1)), mode="edge")
        #kernel = np.array([[x**2+y**2 for x in np.linspace(-1, 1, num=self.terrain_size+1)] for y in np.linspace(-1, 1, num=self.terrain_size+1)])
        terrain = normalize(noise)**exp
        image = pick_color(terrain, BLUE_MARBLE)
        
        self.terrain = terrain*self.terrain_height#*kernel

        t = time.process_time()
        tile = self.martini.create_tile(self.terrain.astype(np.float32))
        #print("martini:", time.process_time() - t)
        
        vertices, faces = tile.get_mesh(1.0)
        vertices = martini.rescale_positions(vertices, self.terrain)
        vertices = vertices[:,[0,2,1]]
        faces = faces.reshape(-1, 3)
        colors = np.array([image[round(x), round(z)] for z, _, x in vertices]).astype(float)/255.0

        mesh = viz.Mesh.from_faces(vertices, faces, colors)
        scene.add(mesh)

        # camera
        x = self.terrain_size//2
        z = self.terrain_size//2
        y = self.terrain_height*2

        fov = 45 #180/min(self.shape)

        scene.camera_position = (x, y, z)
        scene.up_vector = (0, 1, 0)
        scene.camera_target = (0, 0, -1)
        scene.camera_matrix = pr.Matrix44.perspective_projection(fov, 1., 0.1, self.terrain_size)

        # position
        player = np.array([random.integers(0, d) for d in self.shape])

        # targets
        positions = []
        
        side = 12

        tree_line = np.logical_and(terrain >= 0.5, terrain < 0.75)
        target_prob = tree_line.astype(float)/tree_line.sum()

        for z, x in sample_coords((self.terrain_size+1, self.terrain_size+1), self.num_targets, target_prob, random=random):
            y = self.height(x, z)+side
            positions.append((x, y, z))

        """
        shape = (self.terrain_size+1, self.terrain_size+1)
        water = np.logical_and(terrain >= 0.00, terrain < 0.33)
        coast = np.logical_and(terrain >= 0.33, terrain < 0.66)
        hills = np.logical_and(terrain >= 0.66, terrain <= 1.0)

        target_1, = sample_coords(shape, 1, water.astype(float)/water.sum(), random=random)
        target_2, = sample_coords(shape, 1, coast.astype(float)/coast.sum(), random=random)
        target_3, = sample_coords(shape, 1, hills.astype(float)/hills.sum(), random=random)

        positions = [(x, self.height(x, z) + side/2, z) for z, x in [target_1, target_2, target_3]]
        """

        meshes = viz.Mesh.from_boxes(positions, [[side/2]*3]*len(positions), [[255, 0, 0]]*len(positions))
        scene.add(meshes)

        # visibility
        # todo: target might be occluded

        t = time.process_time()

        targets = []
        
        for position in positions:
            visible = False
            closest_direction = None
            closest_distance = np.inf

            # todo: we can ignore certain directions based on angles

            for direction in [(y, x) for y in range(self.shape[0]) for x in range(self.shape[1])]:
                self.look(direction)

                p = self.scene.mvp * pr.Vector4.from_vector3(position, 1.0)        
                p = pr.Vector3(np.array(p[:3]) / p[3])

                if np.all((p >= -1.0) & (p <= 1.0)):
                    visible = True
                else:
                    continue

                distance = np.linalg.norm((0, 0) - p.xy)

                if distance < closest_distance:
                    closest_direction = direction
                    closest_distance = distance

            targets.append(closest_direction)
            #targets.append((0, 0))

            assert visible, f"target at {position} invisible, player {player}"

        #print("targets:", time.process_time() - t)

        return scene, player, targets


    def render(self, mode="rgb_array"):
        self.look(self.position)
        #self.framebuffer.use()
        self.scene.render()
        #frame = np.frombuffer(self.framebuffer.read(components=4), dtype=np.uint8).reshape(*(self.framebuffer.size + (4,)))[::-1]
        frame = self.scene.frame
        image = frame[:,:,:3]

        return image

    def look(self, position):
        eps = 0.1
        pitch_min, pitch_max = 0, -np.pi/2+eps # np.pi/2-eps
        pitch, yaw = np.array(position)/self.shape
        yaw = 2*np.pi*yaw - np.pi/2
        pitch = (((pitch - 0.0) * (pitch_max - pitch_min)) / (1.0 - 0.0)) + pitch_min
        direction = pr.Vector3((np.cos(yaw)*np.cos(pitch), np.sin(pitch), np.sin(yaw)*np.cos(pitch)))
        front = direction.normalized
        self.scene.camera_target = self.scene.camera_position + front

    def observation(self):
        return dict(
            image=self.render(),
            position=self.position
        )

    def height(self, x, z):
        return self.terrain[round(x), round(z)]
