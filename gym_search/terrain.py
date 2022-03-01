import numpy as np
import cairo
import skimage

from gym_search.utils import gaussian_kernel, normalize, clamp
from gym_search.palette import pick_color, EARTH_FEW
from gym_search.noise import fractal_noise_2d


def check_terrain(terrain):
    return 0.0 <= terrain.min() and terrain.max() <= 1.0

def sample_coords(shape, n, p, random=np.random):
    choice = random.choice(np.prod(shape), n, p=p.flatten(), replace=False)
    coords = lambda i: (i//shape[1], i%shape[1])
    return [coords(i) for i in choice]


def basic_terrain(shape, random, size, sigma=1, num_targets=3, num_kernels=1):
    h, w = shape
    kernel = gaussian_kernel(size, sigma=sigma)
    plane = np.zeros((h+size*2,w+size*2))

    for _ in range(num_kernels):
        y, x = random.randint(0, h+size), random.randint(0, w+size)
        plane[y:y+size,x:x+size] += kernel

    terrain = normalize(plane[size:size+h,size:size+w])
    prob = terrain/terrain.sum()
    targets = sample_coords(shape, num_targets, prob, random=random)
    img = np.zeros((*shape, 3), dtype=np.uint8)
    img[:,:,0] = terrain*255

    for y, x in targets:
        img[y,x] = (255, 255, 0)

    return img, targets


def realistic_terrain(shape, random, exp=2):
    # https://jackmckew.dev/3d-terrain-in-python.html
    # https://www.redblobgames.com/maps/terrain-from-noise/
    # http://www-cs-students.stanford.edu/~amitp/game-programming/polygon-map-generation/
    # http://devmag.org.za/2009/05/03/poisson-disk-sampling/
    
    height, width = shape
    noise = fractal_noise_2d(shape, periods=(4, 4), octaves=4, seed=random.randint(9999))
    terrain = normalize(noise**exp)
    img = pick_color(terrain, EARTH_FEW)

    tree_line = np.logical_and(terrain >= 0.5, terrain < 0.75)
    tree_prob = tree_line.astype(float)/tree_line.sum()
    tree_count = random.randint(50, 250)
    trees = sample_coords(shape, tree_count, tree_prob, random=random)

    for y, x in trees:
        r = random.randint(3, 5)
        y = clamp(y, r, height-r)
        x = clamp(x, r, width-r)
        rr, cc = skimage.draw.disk((y, x), r)
        img[rr, cc] = (0, 63, 0)

    target_prob = tree_prob
    target_count = random.randint(0, 10)
    target_pos = sample_coords(shape, target_count, target_prob, random=random)

    for y, x in targets:
        r = random.randint(2, 4)
        y = clamp(y, r, height-r)
        x = clamp(x, r, width-r)
        rr, cc = skimage.draw.disk((y, x), r)
        img[rr, cc] = (255, 0, 0)

    return img, targets