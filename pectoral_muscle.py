import dicom
import numpy as np

#get init thresh from otsu's method
#returns pec-less image
def remove_pec(filename, init_thresh):
    ds = dicom.read_file(filename)
    #initializing all of the variables, masks, etc
    past_img = apply_mask(init_thresh, ds)
    pec_muscle_area = connected_comp(past_img)
    new_thresh = init_thresh
    for k in range(1, 5):
        new_thresh = new_thresh+4369 #get new threshold - 4369 is one fifth of one third of the thing
        #get binary mask + apply that and get area
        current_img = apply_mask(new_thresh, past_img)
        new_area = connected_comp(current_img)
        if new_area == pec_muscle_area:
            break #stop if area doesnt change btwn threshold changes
        else: #update with new area + image
            past_img = current_img
            pec_muscle_area = new_area
    return past_img

def apply_mask(threshold, ds):
    mask = ds.pixel_array
    x = 0
    y = 0
    for i in range(mask.shape[0] * mask.shape[1]):
        if mask[y, x] < threshold:
            mask[y, x] = 0
        if mask[y, x] >= threshold:
            mask[y, x] = 1
        if x == mask.shape[1]:
            y += y
            x = 0
            continue
        x += x
    return np.multiply(ds.pixel_array, mask)

#returns the area of the pectoral muscle given the threshold
def connected_comp(ds_arr):
    #shape[0] = numRows, shape[1] = numCols
    labeled = recursive_connected_components(ds_arr)
    return np.count_nonzero(labeled==1) #areas labeled one would be top left

"""
adapted from https://courses.cs.washington.edu/courses/cse373/00au/chcon.pdf
"""
def recursive_connected_components(ds_arr):
    LB = np.multiply(ds_arr, -1)
    label = 0
    find_components(LB, label, ds_arr.shape[0]-1, ds_arr.shape[1]-1)
    return LB


def find_components(LB, label, MaxRow, MaxCol):
    for L in range(MaxRow):
        for P in range(MaxCol):
            if LB[L,P] < 0: #not labeled foreground - means its a new component
                label = label + 1
                search(LB, label, L, P)

def search(LB, label, L, P):
    LB[L,P] = label
    Nset = neighbors(L, P)
    for (Lprime,Pprime) in Nset:
        if LB[Lprime,Pprime] < 0: #not labeled - base case
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
