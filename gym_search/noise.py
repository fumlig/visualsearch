import numpy as np
import opensimplex


def simplex_noise_2d(x, y):
    return opensimplex.noise2array(x, y)


def fractal_noise_2d(shape, periods=(1, 1), octaves=1, persistence=0.5, lacunarity=2, seed=None):
    if seed is not None:
        opensimplex.seed(seed)

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