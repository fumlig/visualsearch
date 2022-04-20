import numpy as np
from skimage import draw

from gym_search.utils import fractal_noise_2d, normalize, sample_coords
from gym_search.shapes import Box
from gym_search.palette import add_with_alpha, pick_color, EARTH_TOON
from gym_search.envs.search import SearchEnv


class TerrainEnv(SearchEnv):
    
    def __init__(
        self,
        shape=(16, 16),
        view=(64, 64),
        num_targets=3,
        target_size=8,
        num_distractors=50,
        distractor_size=4,

    ):
        super().__init__(shape, view, False)

        assert target_size < min(shape)
        assert distractor_size < min(shape)

        self.num_targets = num_targets
        self.target_size = target_size
        self.num_distractors = num_distractors
        self.distractor_size = distractor_size
    

    def generate(self, seed):
        # https://jackmckew.dev/3d-terrain-in-python.html
        # https://www.redblobgames.com/maps/terrain-from-noise/
        # http://www-cs-students.stanford.edu/~amitp/game-programming/polygon-map-generation/
        # http://devmag.org.za/2009/05/03/poisson-disk-sampling/
        
        random = np.random.default_rng(seed)

        shape = self.scale(self.shape)
        height, width = shape
        exp = random.uniform(0.5, 5)
        noise = fractal_noise_2d(shape, periods=(4, 4), octaves=4, seed=seed)
        terrain = normalize(noise)**exp
        image = pick_color(terrain, EARTH_TOON)

        tree_line = np.logical_and(terrain >= 0.5, terrain < 0.75)
        target_prob = tree_line.astype(float)/tree_line.sum()

        for y, x in sample_coords(shape, self.num_distractors, target_prob, random=random):
            y, x = np.clip((y, x), (0, 0), (height-self.distractor_size, width-self.distractor_size))
            rr, cc = draw.disk((y, x), self.distractor_size, shape=shape)
            image[rr, cc] = (0, 63, 0)

        targets = []

        for y, x in sample_coords(shape, self.num_targets, target_prob, random=random):
            y, x = np.clip((y, x), (0, 0), (height-self.target_size, width-self.target_size))
            coords = tuple(draw.rectangle((y, x), extent=(self.target_size, self.target_size), shape=shape))
            image[coords] = (255, 0, 0)
            targets.append(Box(y, x, self.target_size, self.target_size))

        return image, targets
