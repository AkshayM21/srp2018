"""
extra, previously written but currently unused methods from the pectoral muscle file
"""

import pydicom
import numpy as np
import config
import scipy
from scipy import ndimage
import pectoral_muscle


def canny_remove(filename, init_thresh):
    from skimage import feature
    ds = pydicom.dcmread(filename)

    pixel_array = ds.pixel_array.copy()
    subtracted_arr = get_subtracted_arr(pixel_array, init_thresh)
    print(subtracted_arr)
    multiplier = (65536-init_thresh)/256
    print(multiplier)
    divided_arr = np.divide(subtracted_arr, multiplier)
    print(divided_arr)
    mask = get_canny_mask(pixel_array, init_thresh)
    print(mask)


def get_subtracted_arr(image, subtractor):
    x = 0
    y = 0
    for _ in range(image.shape[0] * image.shape[1]):
        if image[y, x] >= subtractor:
            image[y, x] -= subtractor
        else:
            image[y, x] = 0
        if x == image.shape[0] - 1:
            y += 1
            x = 0
            continue
        x += 1
    return image


def get_canny_mask(image, thresh):
    new_img = image > thresh
    x = 0
    y = 0
    for _ in range(new_img.shape[0] * new_img.shape[1]):
        if y < new_img.shape[1]/3 and new_img[y, x]:
            new_img[y, x] = True
        else:
            new_img[y, x] = False
        if x == new_img.shape[0] - 1:
            y += 1
            x = 0
            continue
        x += 1
    """
    for x in range(new_img.shape[0]):
        if new_img[0, x] == 0:
            break
    x1 = 0
    y = 0
    for _ in range(new_img.shape[0]*new_img.shape[1]):
        if x1>x:
            new_img[y, x1] = 1
        else:
            new_img[y, x1] = 0
        if x1 == new_img.shape[0]-1:
            y += 1
            x1 = 0
            continue
        x1 += 1
        """
    return new_img


def get_x_from_eq(x1, y1, x2, y2, y):
    return ((y-y1)*(x2-x1)/(y2-y1))+x1 #the value of x on the line with points at (x1, y1) and (x2, y2)


def get_y_from_eq(x1, y1, x2, y2, x):
    return ((y2-y1)*(x-x1)/(x2-x1))+y1


#get init thresh from otsu's method
#returns pec-less image
def remove_pec(filename, init_thresh):
    ds = pydicom.dcmread(filename)
    #initializing all of the variables, masks, etc
    past_img = apply_mask(init_thresh, ds.pixel_array, ds)
    ds.PixelData = past_img.tostring()
    ds.save_as("C:/Srp 2018/Mass-Training_P_00018_RIGHT_MLO/07-20-2016-DDSM-09956/1-full mammogram images-65391/otsu.dcm")
    pec_muscle_area = connected_comp(past_img, 0, ds)
    new_thresh = init_thresh
    for k in range(1, 5):
        new_thresh = new_thresh+1024 #get new threshold - 4369 is one fifth of one third of the threshold range
        #get binary mask + apply that and get area
        current_img = apply_mask(new_thresh, past_img, ds)
        new_area = connected_comp(current_img, k, ds)
        if new_area == pec_muscle_area:
            break #stop if area doesnt change btwn threshold changes
        else: #update with new area + image
            past_img = current_img
            pec_muscle_area = new_area
    ds.PixelData = past_img.tostring()
    ds.save_as(filename)


def apply_mask(threshold, pixel_array, ds):
    y = 0
    x = 0
    picture = pixel_array.copy()
    mask = pixel_array.copy()
    for i in range(picture.shape[0]*picture.shape[1]):
        if picture[y,x] < threshold:
            flag = False
            if picture[y,x]>0:
                #print(str(picture[y,x])+ " at "+ str(y)+ " , "+str(x))
                #print "The threshold is: " + str(threshold)
                flag = True
            picture[y, x] = 0
            mask[y, x] = 0
        else:
            mask[y, x] = 65535
        if x == picture.shape[1]-1:
            y += 1
            x = 0
            continue
        x += 1
    #ds.PixelData = mask.tostring()
    #ds.save_as("C:/Srp 2018/Mass-Training_P_00001_LEFT_MLO/07-20-2016-DDSM-90988/1-full mammogram images-80834/mask.dcm")
    print("picture is: \n"+ str(picture))
    return picture


#returns the area of the pectoral muscle given the threshold
def connected_comp(ds_arr, iteration, ds):
    #shape[0] = numRows, shape[1] = numCols
    #labeled = recursive_connected_components(ds_arr)
    labeled, nr_objects = ndimage.label(np.divide(ds_arr, 1), structure=ndimage.generate_binary_structure(2, 2))
    print(nr_objects)
    tosave = labeled.copy()
    print("labeled = "+str(labeled))
    y = 0
    x = 0
    for arya in range(tosave.shape[0]*tosave.shape[1]):
        #print("to save1 at" + str(y) + " , " + str(x) + " is" + str(tosave[y, x]));
        tosave[y, x] = tosave[y,x]*(65535/nr_objects)
        #print("to save2 at"+ str(y)+" , "+str(x)+" is"+str(tosave[y,x]));
        if x == tosave.shape[1]-1:
            y += 1
            x = 0
            continue
        x += 1
    ds.PixelData = tosave.tostring()
    ds.save_as(
        "C:/Srp 2018/Mass-Training_P_00018_RIGHT_MLO/07-20-2016-DDSM-09956/1-full mammogram images-65391/conn"+str(iteration)+".dcm")
    return np.count_nonzero(labeled==1) #areas labeled one would be top left


def connected_comp_array(ds_arr, threshold, ds):
    labeled, nr_objects = ndimage.label(np.divide(ds_arr > threshold, 64), structure=ndimage.generate_binary_structure(2, 2))
    tosave = labeled.copy()
    y = 0
    x = 0
    for arya in range(tosave.shape[0] * tosave.shape[1]):
        tosave[y, x] = tosave[y, x] * (65535 / nr_objects)
        if x == tosave.shape[1] - 1:
            y += 1
            x = 0
            continue
        x += 1
    ds.PixelData = tosave.tostring()
    ds.save_as(
        "C:/Srp 2018/Mass-Training_P_00018_RIGHT_MLO/07-20-2016-DDSM-09956/1-full mammogram images-65391/conn.dcm")
    return labeled  # areas labeled one would be top left


def watershed(threshold, pixel_array, ds):
    from skimage import feature
    from skimage import morphology
    y = 0
    x = 0
    image = pixel_array > threshold
    distance = ndimage.distance_transform_edt(image)
    local_maxi = feature.peak_local_max(distance, indices=False, footprint=np.ones((3, 3)), labels=image)
    markers, num_markers = morphology.label(local_maxi, return_num=True)
    print(num_markers)
    labels_ws = morphology.watershed(-distance, markers, mask=image)
    for arya in range(labels_ws.shape[0]*labels_ws.shape[1]):
        #print("to save1 at" + str(y) + " , " + str(x) + " is" + str(tosave[y, x]));
        labels_ws[y, x] = labels_ws[y,x]*(65535/num_markers)
        #print("to save2 at"+ str(y)+" , "+str(x)+" is"+str(tosave[y,x]));
        if x == labels_ws.shape[1]-1:
            y += 1
            x = 0
            continue
        x += 1
    ds.PixelData = labels_ws.tostring()
    ds.save_as(
        "C:/Srp 2018/Mass-Training_P_00018_RIGHT_MLO/07-20-2016-DDSM-09956/1-full mammogram images-65391/watershed.dcm")

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
    Nset = pectoral_muscle.neighbors(L, P, LB)
    for (Lprime,Pprime) in Nset:
        if LB[Lprime,Pprime] < 0 and (Lprime, Pprime) not in config.alreadyseen: #not labeled - base case
            config.alreadyseen.append((Lprime, Pprime))
            search(LB, label, Lprime, Pprime)
