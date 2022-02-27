import numpy as np
import cv2 as cv
import enum

from gym_search.utils import gaussian_kernel, softmax
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
    OCEAN = (1.0, 0.0, 0.0)
    BEACH = (255, 255, 102)
    SCORCHED = (255, 204, 102)
    BARE = (102, 153, 0)
    TUNDRA = (255, 204, 153)
    SNOW = (255, 255, 255)
    TEMPERATE_DESERT = (204, 102, 0)
    SHRUBLAND = (102, 153, 0)
    TAIGA = (153, 51, 51)
    GRASSLAND = (51, 153, 51)
    TEMPERATE_DECIDUOUS_FOREST = (0, 204, 102)
    TEMPERATE_RAIN_FOREST = (51, 102, 0)
    SUBTROPICAL_DESERT = (255, 255, 153)
    TROPICAL_SEASONAL_FOREST = (51, 153, 102)
    TROPICAL_RAIN_FOREST = (102, 102, 51)


def realistic_terrain(shape, random):
    # https://www.redblobgames.com/maps/terrain-from-noise/
    # http://www-cs-students.stanford.edu/~amitp/game-programming/polygon-map-generation/
    # http://devmag.org.za/2009/05/03/poisson-disk-sampling/


    rng1 = np.random.default_rng(12345)
    rng2 = np.random.default_rng(54321)
    gen1 = OpenSimplex(rng1.integers(10))
    gen2 = OpenSimplex(rng2.integers(10))

    exp = 6.97
    e_oct = [1.00, 0.50, 0.25, 0.13, 0.06, 0.03]
    m_oct = [1.00, 0.75, 0.33, 0.33, 0.33, 0.50]    

    h, w = shape
    y, x = np.arange(h), np.arange(w)
    ny, nx = y/h - 0.5, x/w - 0.5 

    e = np.zeros(shape)
    m = np.zeros(shape)

    for i, e_i in enumerate(e_oct):
        e += e_i*gen1.noise2array(ny*(2**i), nx*(2**i))
    
    for i, m_i in enumerate(m_oct):
        m += m_i*gen2.noise2array(ny*(2**i), nx*(2**i))

    e /= np.sum(e_oct)
    m /= np.sum(m_oct)

    e = np.sign(e)*(np.abs(e))**exp

    terrain = np.empty((*shape, 3), dtype=np.uint8)

    arid = (240, 219, 204)
    humid = (158, 227, 171)

    e = (e+1)/3
    m = (m+1)/3

    print(e.max(), e.min())

    terrain[:,:,0] = e*arid[0] + m*humid[0]    
    terrain[:,:,1] = e*arid[1] + m*humid[1]    
    terrain[:,:,2] = e*arid[2] + m*humid[2]    

    return terrain