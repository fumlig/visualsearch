import numpy as np

from gym_search.envs.search import SearchEnv

from typing import Tuple
from torch.utils.data import Dataset

class DatasetEnv(SearchEnv):
    """
    NOT UPDATED. COULD BE USEFUL FOR USING AN OBJECT LOCALIZATION DATASET.

    Environment for testing search agents on object localization datasets.
    """

    def __init__(
        self,
        shape: Tuple[int, int],
        view: Tuple[int, int],
        dataset: Dataset,
        **kwargs
    ):
        """
        Initialize dataset environment.
        
        shape: Shape of search space.
        view: Shape of image observations (shape*view should be equal to the size of images in the dataset).
        dataset: Object localization dataset.
        **kwargs: Passed to super constructor.
        """

        super().__init__(shape, view, False, **kwargs)
        
        self.dataset = dataset

        self.action_space = super().action_space
        self.observation_space = super().observation_space

    def _generate(self, seed):
        random = np.random.default_rng(seed)
        idx = random.choice(len(self.dataset))
        image, targets = self.dataset[idx]
        position = np.array([random.integers(0, d) for d in self.shape])

        return image, [(*pos, *shape) for pos, shape in targets], position

