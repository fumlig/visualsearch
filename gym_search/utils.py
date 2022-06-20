import numpy as np
import itertools


def to_point(i, w):
    return i//w, i%w

def to_index(x, y, w):
    return y*w + x

def softmax(a):
    return np.exp(a)/np.sum(np.exp(a))

def normalize(a):
    a -= a.min()
    a /= a.max()
    return a

def clamp(x, lo, hi):
    return max(min(x, hi), lo)

def lerp(a, b, x):
    return a + x * (b-a)

def fade(t):
    return 6 * t**5 - 15 * t**4 + 10 * t**3

def gradient(h, x, y):
    vectors = np.array([[0,1],[0,-1],[1,0],[-1,0]])
    g = vectors[h%4]
    return g[:,:,0] * x + g[:,:,1] * y

def perlin(x, y, random):
    p = np.arange(256, dtype=int)
    random.shuffle(p)
    p = np.stack([p,p]).flatten()

    xi = x.astype(int)
    yi = y.astype(int)

    xf = x - xi
    yf = y - yi

    u = fade(xf)
    v = fade(yf)

    n00 = gradient(p[p[xi]+yi],xf,yf)
    n01 = gradient(p[p[xi]+yi+1],xf,yf-1)
    n11 = gradient(p[p[xi+1]+yi+1],xf-1,yf-1)
    n10 = gradient(p[p[xi+1]+yi],xf-1,yf)

    x1 = lerp(n00,n10,u)
    x2 = lerp(n01,n11,u)

    return lerp(x1,x2,v)

def gaussian_kernel(size, sigma=1):
    ax = np.linspace(-(size - 1) / 2, (size - 1) / 2, size)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)

def sample_coords(shape, n, p, random=np.random):
    choice = random.choice(np.prod(shape), n, p=p.flatten(), replace=False)
    return [to_point(i, shape[1]) for i in choice]

def manhattan_dist(p1, p2):
    p12 = np.array(p2) - np.array(p1)
    return np.sum(np.abs(p12))

def euclidean_dist(p1, p2):
    return np.linalg.norm(np.array(p2) - np.array(p1))

def travel_dist(points, dist_func=manhattan_dist):
    # naïve TSP, maybe make something better if it feels necessary
    min_dist = np.inf

    for perm in itertools.permutations(points):
        dist = 0.0
        
        for i in range(len(perm)-1):
            p1 = np.array(perm[i])
            p2 = np.array(perm[i+1])
            dist += dist_func(p1, p2)

            if dist > min_dist:
                break
        
        min_dist = min(dist, min_dist)

    return min_dist

"""
import opensimplex

def simplex_noise_2d(x, y):
    return opensimplex.noise2array(x, y)

@functools.lru_cache(maxsize=1024)
def fractal_noise_2d(shape, periods=(1, 1), octaves=1, persistence=0.5, lacunarity=2, seed=None):
    if seed is not None:
        opensimplex.seed(int(seed))

    h, w = shape
    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1

    for _ in range(octaves):
        y, x = np.arange(h)*frequency*periods[0]/h, np.arange(w)*frequency*periods[1]/w
        noise += amplitude * simplex_noise_2d(x, y)
        frequency *= lacunarity
        amplitude *= persistence
    return noise

    # todo: can we do a sum reduce? if it seems necessary...
"""

import pyfastnoisesimd as fns

def fractal_noise_2d(shape, periods=(1, 1), octaves=4, persistence=0.45, lacunarity=2.1, seed=None):
    perlin = fns.Noise(seed, numWorkers=4)
    perlin.frequency = 0.0025
    perlin.noiseType = fns.NoiseType.SimplexFractal
    perlin.fractal.octaves = octaves
    perlin.fractal.lacunarity = lacunarity
    perlin.fractal.gain = persistence
    perlin.perturb.perturbType = fns.PerturbType.NoPerturb

    noise = perlin.genAsGrid(shape)

    return noise