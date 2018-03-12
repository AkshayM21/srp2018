import pydicom
import numpy as np
import scipy
from scipy import ndimage


def canny_remove(filename):
    from skimage import feature
    ds = pydicom.dcmread(filename)

    pixel_array = ds.pixel_array.copy()

    edges = feature.canny(pixel_array, sigma=0)

    #print(edges)

    y = 0
    x = edges.shape[0]-1
    for _ in range(edges.shape[0]*edges.shape[1]):
        if edges[y, x]:
            break
        if x == 0:
            y += 1
            x = edges.shape[0]-1
            continue
        x -= 1

    y1 = y
    x1 = x

    #print(str(x1)+" , "+str(y1))

    y = 2*edges.shape[1]/3
    x = 0
    for _ in range(edges.shape[0]*edges.shape[1]/3):
        if edges[y, x]:
            break
        if x == edges.shape[0]-1:
            y += 1
            x = 0
            continue
        x += 1
    y2 = y
    x2 = x

   # if x2 > edges.shape[0]/2:
    #    x2 = edges.shape[0] - x2

    #print(str(x2) + " , " + str(y2))

    new_arr = ds.pixel_array.copy()
    for y in range(min(get_y_from_eq(x1, y1, x2, y2, 0), 224)):
        for x in range(min(get_x_from_eq(x1, y1, x2, y2, y), 224)):
            new_arr[y, x] = 0
    ds.PixelData = new_arr.tostring()
    ds.save_as(filename)


def get_x_from_eq(x1, y1, x2, y2, y):
    return ((y-y1)*(x2-x1)/(y2-y1))+x1 #the value of x on the line with points at (x1, y1) and (x2, y2)


def get_y_from_eq(x1, y1, x2, y2, x):
    return ((y2-y1)*(x-x1)/(x2-x1+1))+y1


def neighbors(L, P, LB):
    maxRow = LB.shape[0]
    maxCol = LB.shape[1]
    neighbor = []
    if L-1 >= 0 and P-1 >= 0:
        neighbor.append((L-1,P-1))
    if L-1 >= 0:
        neighbor.append((L-1,P))
    if L+1 < maxRow and P-1 >= 0:
        neighbor.append((L+1, P-1))
    if P-1 >= 0:
        neighbor.append((L, P-1))
    if P+1 < maxCol:
        neighbor.append((L, P+1))
    if L+1 < maxRow and P-1 >= 0:
        neighbor.append((L+1, P-1))
    if L+1 < maxRow:
        neighbor.append((L+1, P))
    if L+1 < maxRow and P+1 < maxCol:
        neighbor.append((L+1, P+1))
    return neighbor
