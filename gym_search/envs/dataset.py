import numpy as np
from skimage import draw

from gym_search.utils import gaussian_kernel, normalize, sample_coords
from gym_search.shapes import Box
from gym_search.palette import add_with_alpha
from gym_search.envs.search import SearchEnv, Action


class DatasetEnv(SearchEnv):
    
    def __init__(
        self,
        shape,
        view,
        dataset,
        **kwargs
    ):
        super().__init__(shape, view, False, **kwargs)
        
        self.dataset = dataset

        self.action_space = super().action_space
        self.observation_space = super().observation_space

    def generate(self, seed):
        random = np.random.default_rng(seed)
        idx = random.choice(len(self.dataset))
        image, targets = self.dataset[idx]
        return image, [Box(*pos, *shape) for pos, shape in targets]

