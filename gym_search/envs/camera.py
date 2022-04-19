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
        terrain_size=1024,
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
        self.scene = viz.Scene(background=(0.75, 0.75, 1.0, 1.0), size=self.view)


    def generate(self, seed):
        scene = self.scene
        
        scene.clear()
        random = np.random.default_rng(seed)

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
        colors = np.array([image[round(x), round(z)] for z, _, x in vertices]).astype(float)/255.0

        mesh = viz.Mesh.from_faces(vertices, faces, colors)
        scene.add(mesh)

        # targets
        targets = []
        tree_line = np.logical_and(terrain >= 0.5, terrain < 0.75)
        target_prob = tree_line.astype(float)/tree_line.sum()

        side = 8

        for z, x in sample_coords((self.terrain_size+1, self.terrain_size+1), self.num_targets, target_prob, random=random):
            y = self.height(x, z)+side
            targets.append((x, y, z))
        
        meshes = viz.Mesh.from_boxes(targets, [[side/2]*3]*len(targets), [[255, 0, 0]]*len(targets))
        scene.add(meshes)

        # camera
        x = self.terrain_size//2
        z = self.terrain_size//2
        y = self.terrain_height*4 # self.height(x, z)

        scene.camera_position = (x, y, z)
        scene.up_vector = (0, 1, 0)
        scene.camera_target = (0, 0, -1)
        scene.camera_matrix = pr.Matrix44.perspective_projection(360/self.shape[0], 1., 0.1, 1000.)


        return scene, targets


    def render(self, mode="rgb_array"):
        eps = 0.1
        pitch, yaw = self.position/self.shape
        yaw = 2*np.pi*yaw - np.pi/2
        pitch = np.clip(np.pi*(0.5-pitch), -np.pi/2+eps, np.pi/2-eps)
        direction = pr.Vector3((np.cos(yaw)*np.cos(pitch), np.sin(pitch), np.sin(yaw)*np.cos(pitch)))
        front = direction.normalized
        self.scene.camera_target = self.scene.camera_position + front

        self.scene.render()
        img = self.scene.frame[:,:,:3]

        return img

    def plot(self, ax, overlay=True, position=None):
        if position is not None:
            _position = self.position
            self.position = position

        img = self.render()

        if position is not None:
            self.position = _position

        ax.imshow(img)

        if overlay:
            ax.set_yticks([0, self.view[0]-1])
            ax.set_xticks([0, self.view[1]-1])
        else:
            ax.set_yticks([])
            ax.set_xticks([])
            

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
