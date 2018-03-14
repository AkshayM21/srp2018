import numpy

def constructPatch(picture_array, mpixelx, mpixely, radius):
    coords_correct = []
    top_left_y = mpixely-int(radius/2)
    top_left_x = mpixelx-int(radius/2)
    #coords_correct.append(mpixely - radius/2, mpixelx - radius/2)
    y = top_left_y
    x = top_left_x
    for i in range(radius*radius):
        if x == top_left_x + radius:
            x = top_left_x
            y  = y+ 1
        else:
            x = x + 1
        if picture_array[y, x] != 0:
            coords_correct.append((y, x))
    return(coords_correct)



"""

patch_check:
hold all the coordinates (y,x)
"""
def correct(coords_correct, patch_check, answer):
    in_patch = 0
    true_positives = 0
    false_positives = 0
    corrects = 0
    incorrects = 0
    y =0
    x = 0

    for i in get_coordinate_tuple(patch_check):
        for j in coords_correct:
            if i == j:
                in_patch = in_patch + 1

    if in_patch >= len(coords_correct)*0.20 and answer == True:
        corrects = corrects + 1
        true_positives+=1
    if in_patch >= len(coords_correct)*0.20 and answer == False:
        incorrects = incorrects + 1
    if in_patch < len(coords_correct)*0.20 and answer == True:
        incorrects = incorrects + 1
        false_positives += 1
    if in_patch < len(coords_correct)*0.20 and answer == False:
        corrects = corrects + 1

    return corrects, incorrects, true_positives, false_positives

def get_coordinate_tuple(blockinfo):
    topleftx = blockinfo[0]
    toplefty = blockinfo[1]
    diameter = blockinfo[2]
    coordinate_list = []
    x = topleftx
    y = toplefty
    for _ in range(diameter**2):
        coordinate_list.append((y, x))
        if x==topleftx+diameter-1:
            x = topleftx
            y+=1
            continue
        x+=1
    return  coordinate_list




