import numpy as np
import cv2 as cv
from gym_search.terrain import realistic_terrain
from tqdm import tqdm


img = realistic_terrain((1024, 1024), np.random)

cv.imshow("terrain", img)
cv.waitKey(0)