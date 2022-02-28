import numpy as np
import cv2 as cv
import enum

from gym_search.utils import gaussian_kernel, softmax, lerp
from opensimplex import OpenSimplex


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

def check_terrain(terrain):
    return 0.0 <= terrain.min() and terrain.max() <= 1.0

def normalize_terrain(terrain):
    return terrain * 1.0/terrain.max()


class Biome(enum.Enum):
    DESERT = (0, 255, 255)
    PLAINS = (0, 255 , 0)
    TUNDRA = (127, 127, 127)
    SAVANNA = (255, 127, 127)
    SHRUBLAND = (0, 127, 0)
    TAIGA = (127, 127, 0)
    FOREST = (127, 255, 127)
    SWAMP = (0, 127, 127)
    SEASONAL_FOREST = (127, 255, 127)
    RAIN_FOREST = (10, 255, 10)


def realistic_terrain(shape, random):
    # https://jackmckew.dev/3d-terrain-in-python.html

    # https://www.redblobgames.com/maps/terrain-from-noise/
    # http://www-cs-students.stanford.edu/~amitp/game-programming/polygon-map-generation/
    # http://devmag.org.za/2009/05/03/poisson-disk-sampling/

    gen1 = OpenSimplex(random.randint(99999))
    gen2 = OpenSimplex(random.randint(99999))

    exp = 5
    e_oct = [1.00, 0.50, 0.25, 0.13, 0.06, 0.03]
    m_oct = [1.00, 0.75, 0.33, 0.33, 0.33, 0.50]    

    h, w = shape
    y, x = np.arange(h), np.arange(w)
    ny, nx = y/h - 0.5, x/w - 0.5 

    e = np.zeros(shape)
    m = np.zeros(shape)

    for i, e_i in enumerate(e_oct):
        e += e_i*gen1.noise2array(ny*(2**i), nx*(2**i))/2+0.5
    
    for i, m_i in enumerate(m_oct):
        m += m_i*gen2.noise2array(ny*(2**i), nx*(2**i))/2+0.5

    e /= np.sum(e_oct)
    m /= np.sum(m_oct)
    e = np.sign(e)*(np.abs(e))**exp

    e /= e.max()
    m /= m.max()

    return e
