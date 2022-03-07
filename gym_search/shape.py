import numpy as np

class Rect:
    def __init__(self, y, x, h, w):
        self.y, self.x = y, x
        self.h, self.w = h, w

    def get_pos(self):
        return self.y, self.x

    def set_pos(self, p):
        self.y, self.x = p

    def get_shape(self):
        return self.h, self.w
    
    def set_shape(self, s):
        self.h, self.w = s

    pos = property(get_pos, set_pos)
    shape = property(get_shape, set_shape)

    def area(self):
        return self.h*self.w

    def corners(self):
        return self.y, self.x, self.y + self.h, self.x + self.w

    def contains(self, p):
        y0, x0, y1, x1 = self.corners()
        y, x = p
        return y0 <= y and y <= y1 and x0 <= x and x <= x1

    def union(self, rect):
        return self.area() + rect.area() - self.intersection(rect)

    def intersection(self, rect):
        y0 = max(self.y, rect.y)
        x0 = max(self.x, rect.x)
        y1 = min(self.y+self.h, rect.y+rect.h)
        x1 = min(self.x+self.w, rect.x+rect.w)
        return max(0, x1-x0) * max(0, y1-y0)

    def overlap(self, rect):
        return self.intersection(rect)/self.union(rect)