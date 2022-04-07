import numpy as np
from skimage import draw
from functools import lru_cache

from gym_search.utils import gaussian_kernel, normalize, clamp, sample_coords
from gym_search.palette import pick_color, EARTH_TOON, BLUE_MARBLE
from gym_search.noise import fractal_noise_2d
from gym_search.shapes import Box


class Generator:
    def __init__(self, training_samples, test_samples):
        pass

    def training_sample(self):
        raise NotImplementedError

    def test_sample(self):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError

    def seed(self, seed=None):
        self.random = np.random.default_rng(seed)


class GaussianGenerator(Generator):
    def __init__(self, shape, num_targets, target_size, num_kernels, kernel_size, sigma=1):
        assert target_size < min(shape)
        assert kernel_size < min(shape)

        self.shape = shape
        self.num_targets = num_targets
        self.target_size = target_size
        self.num_kernels = num_kernels
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.seed()

    def sample(self):
        h, w = self.shape
        kernel = gaussian_kernel(self.kernel_size, sigma=self.sigma)
        terrain = np.zeros(self.shape)

        for _ in range(self.num_kernels):
            y, x = self.random.integers(0, h-self.kernel_size), self.random.integers(0, w-self.kernel_size)
            terrain[y:y+self.kernel_size,x:x+self.kernel_size] += kernel

        terrain = normalize(terrain)
        prob = terrain/terrain.sum()
        targets = sample_coords(self.shape, self.num_targets, prob, random=self.random)
        img = np.zeros((*self.shape, 3), dtype=np.uint8)
        img[:,:,2] = terrain*255

        for y, x in targets:
            r = self.target_size//2
            y = clamp(y, r, h-r)
            x = clamp(x, r, w-r)
            rr, cc = draw.rectangle((y, x), extent=(self.target_size, self.target_size), shape=self.shape)
            img[rr, cc] = (255, 127, 0)

        return img, [Box(*t, self.target_size, self.target_size) for t in targets]


class TerrainGenerator(Generator):
    def __init__(self, shape, num_targets, num_distractors, max_terrains=2**64-1):
        self.shape = shape
        self.num_targets = num_targets
        self.num_distractors = num_distractors
        self.max_terrains = max_terrains
        self.seed()

    def sample(self):
        # https://jackmckew.dev/3d-terrain-in-python.html
        # https://www.redblobgames.com/maps/terrain-from-noise/
        # http://www-cs-students.stanford.edu/~amitp/game-programming/polygon-map-generation/
        # http://devmag.org.za/2009/05/03/poisson-disk-sampling/
        
        height, width = self.shape
        seed = self.random.integers(self.max_terrains)
        terrain = self.terrain(seed)
        image = self.image(terrain)

        for y, x in self.distractors(terrain):
            r = self.random.integers(3, 5)
            y = clamp(y, r, height-r)
            x = clamp(x, r, width-r)
            rr, cc = draw.disk((y, x), r, shape=self.shape)
            image[rr, cc] = (0, 63, 0)

        targets = []

        for y, x in self.targets(terrain):
            size = self.random.integers(5, 10)
            rect = Box(clamp(y, 0, height-size), clamp(x, 0, width-size), size, size)
            targets.append(rect)
            coords = tuple(draw.rectangle(rect.position, extent=rect.shape, shape=self.shape))
            image[coords] = (255, 0, 0)

        return image, targets

    @lru_cache(maxsize=1024)
    def terrain(self, seed):
        exp = self.random.uniform(0.5, 5)
        noise = fractal_noise_2d(self.shape, periods=(4, 4), octaves=4, seed=seed)
        terrain = normalize(noise)**exp

        return terrain
    
    def image(self, terrain, palette=EARTH_TOON):
        return pick_color(terrain, palette)

    def targets(self, terrain):
        tree_line = np.logical_and(terrain >= 0.5, terrain < 0.75)
        target_prob = tree_line.astype(float)/tree_line.sum()
        return sample_coords(self.shape, self.num_targets, target_prob, random=self.random)

    def distractors(self, terrain):
        tree_line = np.logical_and(terrain >= 0.5, terrain < 0.75)
        tree_prob = tree_line.astype(float)/tree_line.sum()
        return sample_coords(self.shape, self.num_distractors, tree_prob, random=self.random)


class DatasetGenerator(Generator):
    def __init__(self, dataset):
        self.dataset = dataset
        self.shape = dataset[0][0].shape[:2]
        self.seed()
    
    def sample(self):
        idx = self.random.choice(len(self.dataset))
        image, targets = self.dataset[idx]
        return image, [Box(*pos, *shape) for pos, shape in targets]