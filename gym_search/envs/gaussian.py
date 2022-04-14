import numpy as np
from skimage import draw

from gym_search.utils import gaussian_kernel, normalize, sample_coords
from gym_search.shapes import Box
from gym_search.palette import add_with_alpha
from gym_search.envs.search import SearchEnv


class GaussianEnv(SearchEnv):
    
    def __init__(
        self,
        shape=(16, 16),
        view=(64, 64),
        num_targets=3,
        target_size=4,
        num_kernels=3,
        kernel_size=8,

    ):
        super().__init__(shape, view, False)

        assert target_size < min(shape)
        assert kernel_size < min(shape)

        self.num_targets = num_targets
        self.target_size = target_size
        self.num_kernels = num_kernels
        self.kernel_size = kernel_size

    def generate(self, seed):
        random = np.random.default_rng(seed)
        size = self.kernel_size*min(self.view)
        sigma = size/6

        h, w = self.scale(self.shape)
        kernel = gaussian_kernel(size, sigma=sigma)
        canvas = np.zeros((h + 2*size, w + 2*size))
        padding = size//2

        for _ in range(self.num_kernels):
            y, x = random.integers(padding, h + size - padding), random.integers(padding, w + size - padding)
            canvas[y:y+size,x:x+size] += kernel

        terrain = canvas[size:-size,size:-size]
        terrain = normalize(terrain)
        prob = terrain/terrain.sum()
        image = np.zeros((h, w, 3), dtype=np.uint8)
        image[:,:,2] = terrain*255

        targets = []

        for y, x in sample_coords((h, w), self.num_targets, prob, random=random):
            y, x = np.clip((y, x), (0, 0), (h - self.target_size, w - self.target_size))
            targets.append(Box(y, x, self.target_size, self.target_size))
            rr, cc = draw.rectangle((y, x), extent=(self.target_size, self.target_size), shape=(h, w))
            image[rr, cc] = (255, 127, 0)

        return image, targets

