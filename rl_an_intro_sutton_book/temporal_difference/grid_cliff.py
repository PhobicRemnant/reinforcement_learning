import numpy as np
from tqdm import tqdm


def isInBounds(x, y, width, height):
    # your code here
    if (0 <= x < height):
        if (0 <= y < width):
            return True

    return False

def isInCliff(loc):

    cliff = [(3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9), (3, 10)]

    for i in cliff:
        if (i == loc):
            return True
    
    return False

def flattenGrid(width, height):
    """
    Turn a grid with a particular width and height from (x,y) positions to a flatenned vector.

    For example:

    |(0,0) (0,1)| = | 0 1 | = [0 1 2 3]
    |(0,1) (1,1)|   | 2 3 | 

    """
    grid = np.arange(height*width).reshape(height,width)
    return grid

w = 12
h = 4

x = 3
y = 1

print(isInBounds(x,y,w,h))

print(isInCliff((1,10)) )

print(flattenGrid(w,h))


