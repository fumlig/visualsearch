import numpy as np
from skimage import draw

from gym_search.utils import gaussian_kernel, normalize, clamp, sample_coords
from gym_search.palette import pick_color, EARTH_TOON, BLUE_MARBLE
from gym_search.noise import fractal_noise_2d
from gym_search.shapes import Box


def gaussian_generator(seed, shape=(256, 256), num_targets=3, target_size=, num_kernels, kernel_size, sigma=1):
    assert target_size < min(shape)
    assert kernel_size < min(shape)

    random = np.random.default_rng(seed)

    h, w = shape
    kernel = gaussian_kernel(kernel_size, sigma=sigma)
    terrain = np.zeros(shape)

    for _ in range(num_kernels):
        y, x = random.integers(0, h-kernel_size), random.integers(0, w-kernel_size)
        terrain[y:y+kernel_size,x:x+kernel_size] += kernel

    terrain = normalize(terrain)
    prob = terrain/terrain.sum()
    targets = sample_coords(shape, num_targets, prob, random=random)
    img = np.zeros((*shape, 3), dtype=np.uint8)
    img[:,:,2] = terrain*255

    for position in targets:
        position = np.clip(position, (0, 0), ())
        r = target_size//2
        y = clamp(y, r, h-r)
        x = clamp(x, r, w-r)
        rr, cc = draw.rectangle((y, x), extent=(target_size, target_size), shape=shape)
        img[rr, cc] = (255, 127, 0)

    return img, [Box(*t, target_size, target_size) for t in targets]



def terrain_generator(seed, shape, num_targets, num_distractors):
    # https://jackmckew.dev/3d-terrain-in-python.html
    # https://www.redblobgames.com/maps/terrain-from-noise/
    # http://www-cs-students.stanford.edu/~amitp/game-programming/polygon-map-generation/
    # http://devmag.org.za/2009/05/03/poisson-disk-sampling/
    
    height, width = shape

    random = np.random.default_rng(seed)

    exp = random.uniform(0.5, 5)
    noise = fractal_noise_2d(shape, periods=(4, 4), octaves=4, seed=seed)
    terrain = normalize(noise)**exp
    image = pick_color(terrain, EARTH_TOON)

    tree_line = np.logical_and(terrain >= 0.5, terrain < 0.75)
    target_prob = tree_line.astype(float)/tree_line.sum()

    targets = sample_coords(shape, num_targets, target_prob, random=random)
    distractors = sample_coords(shape, num_distractors, target_prob, random=random)

    for y, x in distractors:
        r = random.integers(3, 5)
        y = clamp(y, r, height-r)
        x = clamp(x, r, width-r)
        rr, cc = draw.disk((y, x), r, shape=shape)
        image[rr, cc] = (0, 63, 0)

    targets = []

    for y, x in targets:
        size = random.integers(5, 10)
        rect = Box(clamp(y, 0, height-size), clamp(x, 0, width-size), size, size)
        targets.append(rect)
        coords = tuple(draw.rectangle(rect.position, extent=rect.shape, shape=shape))
        image[coords] = (255, 0, 0)

    return image, targets


def camera_generator(seed):
    pass


def dataset_generator(seed, dataset):
    shape = dataset[0][0].shape[:2]
    random = np.random.default_rng(seed)
    idx = random.choice(len(dataset))
    image, targets = dataset[idx]
    return image, [Box(*pos, *shape) for pos, shape in targets]
