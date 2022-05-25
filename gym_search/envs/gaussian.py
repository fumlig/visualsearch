import numpy as np
import cv2 as cv
from skimage import draw

from gym_search.utils import gaussian_kernel, normalize, sample_coords
from gym_search.shapes import Box
from gym_search.palette import add_with_alpha
from gym_search.envs.search import SearchEnv, Action


class GaussianEnv(SearchEnv):
    
    def __init__(
        self,
        shape=(10, 10),
        view=(64, 64),
        num_targets=3,
        num_kernels=3,
        **kwargs
    ):
        super().__init__(shape, view, False, **kwargs)

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

        position = np.array([random.integers(0, d) for d in self.shape])

        return image, position, targets

    def get_handcrafted_action(self, detect=False):
        if not detect and any([self.visible(target) and not hit for target, hit in zip(self.targets, self.hits)]):
            return 0

        obs = self.observation()
        img = obs["image"]

        sobel_x = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=3)
        sobel_y = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=3)

        dir_x = np.sum(sobel_x)
        dir_y = np.sum(sobel_y)

        actions = sorted([(dir_x, Action.LEFT), (-dir_x, Action.RIGHT), (dir_y, Action.UP), (-dir_y, Action.DOWN)], reverse=True)
        
        for _, action in actions:
            step = self.get_action_step(action)
            position, _ = self.get_next_position(step)
            
            if not tuple(position) in self.visited:
                return action
        
        return self.get_random_action()