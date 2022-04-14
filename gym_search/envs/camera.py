import numpy as np
import moderngl as gl
import pyrr as pr
import simple_3dviz as viz
import pymartini as martini

from gym_search.utils import sample_coords, fractal_noise_2d, normalize
from gym_search.palette import EARTH_TOON, pick_color
from gym_search.envs.search import SearchEnv


class CameraEnv(SearchEnv):

    metadata = {"render.modes": ["rgb_array"]}

    def __init__(
        self,
        shape=(16, 16),
        view=(64, 64),
        terrain_size=256,
        terrain_height=16,
        num_targets=10,
        num_distractors=100,
    ):
        super().__init__(shape, view, True)

        self.terrain_size = terrain_size
        self.terrain_height = terrain_height
        self.num_targets = num_targets
        self.num_distractors = num_distractors
 
        self.martini = martini.Martini(self.terrain_size+1)


    def generate(self, seed):
        random = np.random.default_rng(seed)
        scene = viz.Scene(background=(0.0, 0.0, 0.0, 1.0), size=self.view)

        # terrain
        exp = random.uniform(0.5, 5)
        noise = fractal_noise_2d((self.terrain_size+1, self.terrain_size+1), periods=(4, 4), octaves=4, seed=seed)
        terrain = normalize(noise)**exp
        image = pick_color(terrain, EARTH_TOON)

        self.terrain = terrain*self.terrain_height
        tile = self.martini.create_tile(self.terrain.astype(np.float32))
        vertices, faces = tile.get_mesh(0.5)
        vertices = martini.rescale_positions(vertices, self.terrain)
        vertices = vertices[:,[0,2,1]]
        faces = faces.reshape(-1, 3)
        colors = np.array([image[round(x), round(z)] for x, _, z in vertices]).astype(float)/255.0

        mesh = viz.Mesh.from_faces(vertices, faces, colors)
        scene.add(mesh)

        # targets
        targets = []
        tree_line = np.logical_and(terrain >= 0.5, terrain < 0.75)
        target_prob = tree_line.astype(float)/tree_line.sum()

        side = 1

        for z, x in sample_coords((self.terrain_size+1, self.terrain_size+1), self.num_targets, target_prob, random=random):
            y = self.height(x, z)+side
            targets.append((x, y, z))
        
        meshes = viz.Mesh.from_boxes(targets, [[side/2]*3]*len(targets), [[255, 0, 0]]*len(targets))
        scene.add(meshes)

        # camera
        x = self.terrain_size//2
        z = self.terrain_size//2
        y = self.terrain_height # self.height(x, z)

        scene.camera_position = (x, y, z)
        scene.up_vector = (0, 1, 0)

        return scene, targets


    def render(self, mode="rgb_array"):
        pitch, yaw = 2*np.pi*self.position/self.shape
        pitch = (pitch+np.pi)/2
        direction = pr.Vector3((np.cos(yaw)*np.cos(pitch), np.sin(pitch), np.sin(yaw)*np.cos(pitch)))
        front = direction.normalized
        self.scene.camera_target = self.scene.camera_position + front

        self.scene.render()
        img = self.scene.frame

        return img

    def close(self):
        self.renderer.delete()

    def observation(self):
        return dict(
            image=self.render(),
            position=self.position
        )

    def height(self, x, z):
        return self.terrain[round(x), round(z)]

    def visible(self, target):
        x, y, z = target

        proj = self.scene.camera_matrix
        view = self.scene.mv
    
        position = proj * view * pr.Vector4([x, y, z, 1.0])    
        p = np.array(position[:3]) / position[3]

        return np.all((p >= -1.0) & (p <= 1.0))
