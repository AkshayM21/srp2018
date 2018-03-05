import pydicom
import numpy as np
import config
import scipy
from scipy import ndimage

#get init thresh from otsu's method
#returns pec-less image
def remove_pec(filename, init_thresh):
    ds = pydicom.dcmread(filename)
    #initializing all of the variables, masks, etc
    past_img = apply_mask(init_thresh, ds.pixel_array)
    ds.PixelData = past_img.tostring()
    ds.save_as("C:/Srp 2018/Mass-Training_P_00001_LEFT_MLO/07-20-2016-DDSM-90988/1-full mammogram images-80834/0000001.dcm")
    pec_muscle_area = connected_comp(past_img)
    new_thresh = init_thresh
    for k in range(1, 5):
        new_thresh = new_thresh+4369 #get new threshold - 4369 is one fifth of one third of the threshold range
        #get binary mask + apply that and get area
        current_img = apply_mask(new_thresh, past_img)
        new_area = connected_comp(current_img)
        if new_area == pec_muscle_area:
            break #stop if area doesnt change btwn threshold changes
        else: #update with new area + image
            past_img = current_img
            pec_muscle_area = new_area
    ds.PixelData = past_img.tostring()
    ds.save_as(filename)

def apply_mask(threshold, pixel_array):
    y = 0
    x = 0
    picture = pixel_array.copy()
    for i in range(picture.shape[0]*picture.shape[1]):
        if picture[y,x] < threshold:
            flag = False
            if picture[y,x]>0:
                print(str(picture[y,x])+ " at "+ str(y)+ " , "+str(x))
                print "The threshold is: " + str(threshold)
                flag = True
            picture[y, x] = 0
            if flag:
               print picture[y,x]
        if x == picture.shape[1]-1:
            y += y
            x = 0
            continue
        x += 1
    """
    print(ds)
    print(ds.pixel_array)
    pixels = ds.pixel_array
    mask = np.zeros(pixels.shape)
    x = 0
    y = 0
    for i in range(mask.shape[0] * mask.shape[1]):
        if pixels[y, x] < threshold:
            mask[y, x] = 0
        if pixels[y, x] >= threshold:
            mask[y, x] = 1
        if x == mask.shape[1]-1:
            y += y
            x = 0
            continue
        x += 1
    print(threshold)
    print("mask: ")
    print mask
    print("pixel array: ")
    print pixels
    #ds.PixelData = pixels.tostring()
    #ds.save_as(
     #   "D:/Akshay SRP 2018/Mass-Training_P_00001_LEFT_MLO/07-20-2016-DDSM-90988/1-full mammogram images-80834/mask.dcm")
"""
    print("picture is: \n"+ str(picture))
    return picture

#returns the area of the pectoral muscle given the threshold
def connected_comp(ds_arr):
    #shape[0] = numRows, shape[1] = numCols
    #labeled = recursive_connected_components(ds_arr)
    labeled, nr_objects = ndimage.label(ds_arr)
    return np.count_nonzero(labeled==1) #areas labeled one would be top right




"""
OLD CONNECTED COMPONENTS ALGORITHM
adapted from https://courses.cs.washington.edu/courses/cse373/00au/chcon.pdf
"""
def recursive_connected_components(ds_arr):
    LB = np.multiply(ds_arr, -1)
    print(LB)
    label = 0
    find_components(LB, label, ds_arr.shape[0], ds_arr.shape[1])
    return LB


def find_components(LB, label, MaxRow, MaxCol):
    for L in range(MaxRow):
        for P in range(MaxCol)[::-1]:
            if LB[L,P] < 0: #not labeled foreground - means its a new component
                label = label + 1
                if((L, P) not in config.alreadyseen):
                    config.alreadyseen.append((L, P))
                    search(LB, label, L, P)

def search(LB, label, L, P):
    LB[L,P] = label
    print('new search at ' + str(L) + ', ' + str(P) + ' label of ' + str(LB[L,P]))
    Nset = neighbors(L, P, LB)
    for (Lprime,Pprime) in Nset:
        if LB[Lprime,Pprime] < 0 and (Lprime, Pprime) not in config.alreadyseen: #not labeled - base case
            config.alreadyseen.append((Lprime, Pprime))
            search(LB, label, Lprime, Pprime)

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
