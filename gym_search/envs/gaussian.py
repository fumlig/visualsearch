import numpy as np
from skimage import draw

from gym_search.utils import gaussian_kernel, normalize, sample_coords
from gym_search.shapes import Box
from gym_search.palette import add_with_alpha
from gym_search.envs.search import SearchEnv, Action


class GaussianEnv(SearchEnv):
    
    def __init__(
        self,
        shape=(16, 16),
        view=(64, 64),
        punish_revisit=False,
        reward_closer=False,
        num_targets=3,
        num_kernels=3,
    ):
        super().__init__(shape, view, False, punish_revisit=punish_revisit, reward_closer=reward_closer)

        self.num_targets = num_targets
        self.num_kernels = num_kernels
        self.target_size = 8
        self.kernel_size = int(min(self.shape)*0.75)

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
        image = np.full((h, w, 3), 255, dtype=np.uint8)
        image[:,:,0] -= (terrain*255).astype(np.uint8)
        image[:,:,1] -= (terrain*255).astype(np.uint8)

        targets = []

        for y, x in sample_coords((h, w), self.num_targets, prob, random=random):
            position = np.array((y, x)) // self.view
            y, x = np.clip((y, x), position*self.view, position*self.view + self.view - (self.target_size, self.target_size))

            #targets.append(Box(y, x, self.target_size, self.target_size))
            targets.append(position)
            
            rr, cc = draw.rectangle((y, x), extent=(self.target_size, self.target_size), shape=(h, w))
            image[rr, cc] = (255, 0, 0)

        return image, targets

