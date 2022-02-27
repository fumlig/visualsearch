import numpy as np
import cv2 as cv
from gym_search.terrain import realistic_terrain
from tqdm import tqdm

while True:
    img = realistic_terrain((512, 512), np.random)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    cv.imshow("terrain", img)
    cv.waitKey(0)

