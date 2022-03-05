import numpy as np

from skimage import draw
from functools import lru_cache

from gym_search.utils import gaussian_kernel, normalize, clamp
from gym_search.palette import pick_color, EARTH_TOON
from gym_search.noise import fractal_noise_2d
from gym_search.shape import Rect


def check_terrain(terrain):
    return 0.0 <= terrain.min() and terrain.max() <= 1.0


def sample_coords(shape, n, p, random=np.random):
    choice = random.choice(np.prod(shape), n, p=p.flatten(), replace=False)
    coords = lambda i: (i//shape[1], i%shape[1])
    return [coords(i) for i in choice]


def gaussian_terrain(shape, seed, size, sigma=1, num_targets=3, num_kernels=1):
    random = np.random.default_rng(seed)

    h, w = shape
    kernel = gaussian_kernel(size, sigma=sigma)
    plane = np.zeros((h+size*2,w+size*2))

    for _ in range(num_kernels):
        y, x = random.integers(0, h+size), random.integers(0, w+size)
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


@lru_cache(maxsize=1024)
def realistic_terrain(shape, seed):
    # https://jackmckew.dev/3d-terrain-in-python.html
    # https://www.redblobgames.com/maps/terrain-from-noise/
    # http://www-cs-students.stanford.edu/~amitp/game-programming/polygon-map-generation/
    # http://devmag.org.za/2009/05/03/poisson-disk-sampling/

    random = np.random.default_rng(seed)

    exp = random.uniform(0.1, 10.0)
    height, width = shape
    noise = fractal_noise_2d(shape, periods=(4, 4), octaves=4, seed=seed)
    terrain = normalize(noise)**exp
    img = pick_color(terrain, EARTH_TOON)

    tree_line = np.logical_and(terrain >= 0.5, terrain < 0.75)
    tree_prob = tree_line.astype(float)/tree_line.sum()
    tree_count = random.integers(100, 500)
    trees = sample_coords(shape, tree_count, tree_prob, random=random)

    for y, x in trees:
        r = random.integers(3, 5)
        y = clamp(y, r, height-r)
        x = clamp(x, r, width-r)
        rr, cc = draw.disk((y, x), r)
        img[rr, cc] = (0, 63, 0)

    target_prob = tree_prob
    target_count = random.integers(5, 25)
    target_pos = sample_coords(shape, target_count, target_prob, random=random)
    targets = []

    for y, x in target_pos:
        size = random.integers(2, 4)
        rect = Rect(clamp(y, 0, height-size), clamp(x, 0, width-size), size, size)
        targets.append(rect)
        coords = draw.rectangle(rect.pos, extent=rect.shape)
        img[coords] = (255, 0, 0)

    return img, targets