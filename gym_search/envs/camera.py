import numpy as np
import moderngl as gl
import pyrr as pr
import simple_3dviz as viz
import pymartini as martini

from gym_search.utils import sample_coords, fractal_noise_2d, normalize
from gym_search.palette import EARTH_TOON, pick_color
from gym_search.envs.search import SearchEnv, Action

class CameraEnv(SearchEnv):

    def __init__(
        self,
        shape=(10, 20),
        view=(64, 64),
        terrain_size=2048,
        terrain_height=16,
        num_targets=3,
        num_distractors=100,
        **kwargs,
    ):
        super().__init__(shape, view, True, **kwargs)

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

        # camera
        x = self.terrain_size//2
        z = self.terrain_size//2
        y = self.terrain_height*4 # self.height(x, z)

        fov = 180/min(self.shape)

        scene.camera_position = (x, y, z)
        scene.up_vector = (0, 1, 0)
        scene.camera_target = (0, 0, -1)
        scene.camera_matrix = pr.Matrix44.perspective_projection(fov, 1., 0.1, self.terrain_size)

        # position
        player = np.array([random.integers(0, d) for d in self.shape])

        # targets
        positions = []
        tree_line = np.logical_and(terrain >= 0.5, terrain < 0.75)
        target_prob = tree_line.astype(float)/tree_line.sum()

        side = 8

        for z, x in sample_coords((self.terrain_size+1, self.terrain_size+1), self.num_targets, target_prob, random=random):
            y = self.height(x, z)+side
            positions.append((x, y, z))
        
        meshes = viz.Mesh.from_boxes(positions, [[side/2]*3]*len(positions), [[255, 0, 0]]*len(positions))
        scene.add(meshes)

        # visibility
        # todo: target might be occluded
        targets = []
        
        for position in positions:
            visible = False

            for target in [(y, x) for y in range(self.shape[0]) for x in range(self.shape[1])]:
                self.look(target)
                
                if self.in_frustum(*position):
                    targets.append(target)
                    visible = True
                    break        
            
            assert visible, f"target at {position} invisible, player {player}"

        return scene, player, targets


    def render(self, mode="rgb_array"):
        self.look(self.position)
        self.scene.render()
        img = self.scene.frame[:,:,:3]

        return img

    def look(self, position):
        eps = 0.1
        pitch, yaw = np.array(position)/self.shape
        yaw = 2*np.pi*yaw - np.pi/2
        pitch = np.clip(np.pi*(0.5-pitch), -np.pi/2+eps, np.pi/2-eps)
        direction = pr.Vector3((np.cos(yaw)*np.cos(pitch), np.sin(pitch), np.sin(yaw)*np.cos(pitch)))
        front = direction.normalized
        self.scene.camera_target = self.scene.camera_position + front

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

    def in_frustum(self, x, y, z):
        position = self.scene.mvp * pr.Vector4([x, y, z, 1.0])        
        p = np.array(position[:3]) / position[3]

        return np.all((p >= -1.0) & (p <= 1.0))
