import numpy as np
import opensimplex


def simplex_noise_2d(shape, periods=1, octaves=1, persistence=0.5, lacunarity=2, random=np.random):
    h, w = shape
    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        y, x = np.arange(h)*frequency*periods/h, np.arange(w)*frequency*periods/w
        noise += amplitude * opensimplex.noise2array(x, y)
        frequency *= lacunarity
        amplitude *= persistence
    return noise

"""
reasonably fast...

img = simplex_noise_2d((2048, 2048), periods=4, octaves=4)
plt.imshow(img, cmap="terrain")
"""