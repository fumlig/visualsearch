import numpy as np
from skimage import draw

from gym_search.utils import gaussian_kernel, normalize, clamp, sample_coords
from gym_search.palette import pick_color, EARTH_TOON
from gym_search.noise import fractal_noise_2d
from gym_search.shapes import Rect


class Generator:
    def generate(self):
        raise NotImplementedError

    def seed(self, seed=None):
        self.random = np.random.default_rng(seed)


class GaussianGenerator(Generator):
    def __init__(self, shape, num_targets, num_kernels, size, sigma=1):
        self.shape = shape
        self.num_targets = num_targets
        self.num_kernels = num_kernels
        self.size = size
        self.sigma = sigma
        self.random = np.random.default_rng()

    def generate(self):
        h, w = self.shape
        kernel = gaussian_kernel(self.size, sigma=self.sigma)
        plane = np.zeros((h+self.size*2,w+self.size*2))

        for _ in range(self.num_kernels):
            y, x = self.random.integers(0, h+self.size), self.random.integers(0, w+self.size)
            plane[y:y+self.size,x:x+self.size] += kernel

        terrain = normalize(plane[self.size:self.size+h,self.size:self.size+w])
        prob = terrain/terrain.sum()
        targets = sample_coords(self.shape, self.num_targets, prob, random=self.random)
        img = np.full((*self.shape, 3), 255, dtype=np.uint8)
        img[:,:,1] = img[:,:,1] - terrain*255
        img[:,:,2] = img[:,:,2] - terrain*255

        for y, x in targets:
            img[y,x] = (255, 255, 0)

        return img, [Rect(*t, 1, 1) for t in targets]


class TerrainGenerator(Generator):
    def __init__(self, shape, max_terrains=1024):
        self.shape = shape
        self.max_terrains = max_terrains
        self.random = np.random.default_rng()

    def generate(self):
        # https://jackmckew.dev/3d-terrain-in-python.html
        # https://www.redblobgames.com/maps/terrain-from-noise/
        # http://www-cs-students.stanford.edu/~amitp/game-programming/polygon-map-generation/
        # http://devmag.org.za/2009/05/03/poisson-disk-sampling/
        
        seed = self.random.integers(self.max_terrains)

        exp = self.random.uniform(0.1, 10.0)
        height, width = self.shape
        noise = fractal_noise_2d(self.shape, periods=(4, 4), octaves=4, seed=seed)
        terrain = normalize(noise)**exp
        img = pick_color(terrain, EARTH_TOON)

        tree_line = np.logical_and(terrain >= 0.5, terrain < 0.75)
        tree_prob = tree_line.astype(float)/tree_line.sum()
        tree_count = self.random.integers(100, 500)
        trees = sample_coords(self.shape, tree_count, tree_prob, random=self.random)

        for y, x in trees:
            r = self.random.integers(3, 5)
            y = clamp(y, r, height-r)
            x = clamp(x, r, width-r)
            rr, cc = draw.disk((y, x), r)
            img[rr, cc] = (0, 63, 0)

        target_prob = tree_prob
        target_count = self.random.integers(5, 25)
        target_pos = sample_coords(self.shape, target_count, target_prob, random=self.random)
        targets = []

        for y, x in target_pos:
            size = self.random.integers(2, 4)
            rect = Rect(clamp(y, 0, height-size), clamp(x, 0, width-size), size, size)
            targets.append(rect)
            coords = draw.rectangle(rect.pos, extent=rect.shape)
            img[coords] = (255, 0, 0)

        return img, targets


class DatasetGenerator(Generator):
    def __init__(self, dataset):
        self.dataset = dataset
        self.shape = dataset[0][0].shape[:2]
        self.random = np.random.default_rng()
    
    def generate(self):
        idx = self.random.choice(len(self.dataset))
        image, targets = self.dataset[idx]
        return image, [Rect(*pos, *shape) for pos, shape in targets]