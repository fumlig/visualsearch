import numpy as np


def to_point(i, w):
    return (i//w, i%w)

def to_index(x, y, w):
    return y*w + x

def softmax(a):
    return np.exp(a)/np.sum(np.exp(a))

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