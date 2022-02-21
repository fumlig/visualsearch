import numpy as np
import cv2 as cv
from gym_search.utils import gaussian_kernel


def uniform_terrain(shape, random, c=1.0):
    return np.full(shape, c)

def gaussian_terrain(shape, random, size, sigma=1, n=1):
    h, w = shape
    kernel = gaussian_kernel(size, sigma=sigma)
    terrain = np.zeros((h+size*2,w+size*2))

    for i in range(n):
        y, x = random.randint(0, h+size), random.randint(0, w+size)
        terrain[y:y+size,x:x+size] += kernel

    return normalize_terrain(terrain[size:size+h,size:size+w])

def horizon_terrain():
    # mountains: A*sin(B*x) + C*sin(D*pi*x) + E*sin(F*e*x)
    # tree height
    pass


def check_terrain(terrain):
    return 0.0 <= terrain.min() and terrain.max() <= 1.0

def normalize_terrain(terrain):
    return terrain * 1.0/terrain.max()
