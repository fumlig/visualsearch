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
    OCEAN = enum.auto()
    BEACH = enum.auto()
    SCORCHED = enum.auto()
    BARE = enum.auto()
    TUNDRA = enum.auto()
    TEMPERATE_DESERT = enum.auto()
    SRUBLAND = enum.auto()
    GRASSLAND = enum.auto()
    TEMPERATE_DECIDUOUS_FOREST = enum.auto()
    TEMPERATE_RAIN_FOREST = enum.auto()
    SUBTROPICAL_DESERT = enum.auto()
    TROPICAL_SEASONAL_FOREST = enum.auto()
    TROPICAL_RAIN_FOREST = enum.auto()

def pick_biome(elevation, moisture):
    e = elevation
    m = moisture

    if e < 0.10: return Biome.OCEAN
    if e < 0.12: return Biome.BEACH

    if e > 0.8:
        if m < 0.1: return Biome.SCORCHED
        if m < 0.2: return Biome.BARE
        if m < 0.5: return Biome.TUNDRA
        return Biome.SNOW
     
    if e > 0.5:
        if m < 0.33: return Biome.TEMPERATE_DESERT
        if m < 0.66: return Biome.SRUBLAND
        return Biome.TAIGA

    if e > 0.3:
        if m < 0.16: return Biome.TEMPERATE_DESERT
        if m < 0.50: return Biome.GRASSLAND
        if m < 0.83: return Biome.TEMPERATE_DECIDUOUS_FOREST
        return Biome.TEMPERATE_RAIN_FOREST

    if m < 0.25: return Biome.SUBTROPICAL_DESERT
    if m < 0.33: return Biome.GRASSLAND
    if m < 0.66: return Biome.TROPICAL_SEASONAL_FOREST
    return Biome.TROPICAL_RAIN_FOREST

def realistic_terrain(shape, random):
    # https://www.redblobgames.com/maps/terrain-from-noise/
    # http://www-cs-students.stanford.edu/~amitp/game-programming/polygon-map-generation/

    # http://devmag.org.za/2009/05/03/poisson-disk-sampling/

    rng1 = np.random.default_rng(12345)
    rng2 = np.random.default_rng(54321)
    gen1 = OpenSimplex(rng1.integers(99999))
    gen2 = OpenSimplex(rng2.integers(99999))

    exp = 6.97
    elevation_oct = [1.00, 0.50, 0.25, 0.13, 0.06, 0.03]
    moisture_oct = [1.00, 0.75, 0.33, 0.33, 0.33, 0.50]    

    h, w = shape

    terrain = np.zeros(shape)

    for y in range(h):
        for x in range(w):
            nx, ny = x/w - 0.5, y/h - 0.5
            
            e = sum([e_i*gen1.noise2(nx*2**i, ny*2**i) for i, e_i in enumerate(elevation_oct)])
            e /= sum(elevation_oct)
            e = e**exp

            m = sum([m_i*gen1.noise2(nx*2**i, ny*2**i) for i, m_i in enumerate(moisture_oct)])
            m /= sum(moisture_oct)

            b = pick_biome(e, m)

