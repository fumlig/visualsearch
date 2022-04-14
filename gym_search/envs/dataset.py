import numpy as np
from skimage import draw

from gym_search.utils import gaussian_kernel, normalize, sample_coords
from gym_search.shapes import Box
from gym_search.palette import add_with_alpha
from gym_search.envs.search import SearchEnv


class DatasetEnv(SearchEnv):
    
    def __init__(
        self,
        shape,
        view,
        dataset,
    ):
        super().__init__(shape, view, False)
        
        self.dataset = dataset

    def generate(self, seed):
        random = np.random.default_rng(seed)
        idx = random.choice(len(self.dataset))
        image, targets = self.dataset[idx]
        return image, [Box(*pos, *shape) for pos, shape in targets]

