import numpy as np
import skimage

from gym_search.utils import gaussian_kernel, normalize, clamp, sample_coords
from gym_search.palette import pick_color, EARTH_TOON
from gym_search.noise import fractal_noise_2d
from gym_search.shape import Rect


class Generator:
    def generate(self):
        raise NotImplementedError


class GaussianGenerator(Generator):
    def __init__(self, shape, random, num_targets, num_kernels, size, sigma=1):
        self.shape = shape
        self.random = random
        self.num_targets = num_targets
        self.num_kernels = num_kernels
        self.size = size
        self.sigma = sigma

    def generate(self):
        h, w = self.shape
        kernel = gaussian_kernel(self.size, sigma=self.sigma)
        plane = np.zeros((h+self.size*2,w+self.size*2))

        for _ in range(self.num_kernels):
            y, x = self.random.integers(0, h+self.size), self.random.integers(0, w+size)
            plane[y:y+size,x:x+size] += kernel

        terrain = normalize(plane[size:size+h,size:size+w])
        prob = terrain/terrain.sum()
        targets = sample_coords(shape, num_targets, prob, random=random)
        img = np.full((*shape, 3), 255, dtype=np.uint8)
        img[:,:,1] = img[:,:,1] - terrain*255
        img[:,:,2] = img[:,:,2] - terrain*255

        for y, x in targets:
            img[y,x] = (0, 255, 0)

        return img, [Rect(*t, 1, 1) for t in targets]



class TerrainGenerator(Generator):
    def __init__(self, shape, random):
        self.random = random
        self.shape = shape
    
    def generate(self):
        # https://jackmckew.dev/3d-terrain-in-python.html
        # https://www.redblobgames.com/maps/terrain-from-noise/
        # http://www-cs-students.stanford.edu/~amitp/game-programming/polygon-map-generation/
        # http://devmag.org.za/2009/05/03/poisson-disk-sampling/
        
        exp = self.random.uniform(0.1, 10.0)
        height, width = self.shape
        noise = fractal_noise_2d(self.shape, periods=(4, 4), octaves=4, seed=self.random.randint(9999))
        terrain = normalize(noise)**exp
        img = pick_color(terrain, EARTH_TOON)

        tree_line = np.logical_and(terrain >= 0.5, terrain < 0.75)
        tree_prob = tree_line.astype(float)/tree_line.sum()
        tree_count = self.random.randint(100, 500)
        trees = sample_coords(self.shape, tree_count, tree_prob, random=self.random)

        for y, x in trees:
            r = self.random.randint(3, 5)
            y = clamp(y, r, height-r)
            x = clamp(x, r, width-r)
            rr, cc = skimage.draw.disk((y, x), r)
            img[rr, cc] = (0, 63, 0)

        target_prob = tree_prob
        target_count = self.random.randint(5, 25)
        target_pos = sample_coords(self.shape, target_count, target_prob, random=self.random)
        targets = []

        for y, x in target_pos:
            size = self.random.randint(2, 4)
            rect = Rect(clamp(y, 0, height-size), clamp(x, 0, width-size), size, size)
            targets.append(rect)
            coords = skimage.draw.rectangle(rect.pos, extent=rect.shape)
            img[coords] = (255, 0, 0)

        return img, targets


class DatasetGenerator(Generator):
    def __init__(self, random, dataset):
        self.random = random
        self.dataset = dataset
        self.shape = dataset[0][0].shape
    
    def generate(self):
        idx = self.random.choice(len(self.dataset))
        image, targets = self.dataset[idx]
        return image, targets