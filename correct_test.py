import numpy

def constructPatch(picture_array, mpixelx, mpixely, radius):
	coords_correct = []
	picture_array[mpixely,mpixelx]
	top_left = picture_array[mpixely-radius, mpixelx-radius]
	coords_correct.append(mpixely - radius, mpixelx - radius)
	y = top_left[1]
	x = top_left[0]
	for i in range(radius*radius):
		if x == top_left[0] + radius:
			x = top_left[0]
			y + 1 = y
		if x != top_left[0] + radius:
			x + 1 = x

		coords_correct.append(y, x)
	return(coords_correct)



"""

patch_check:
hold all the coordinates (y,x)
"""
def correct(probability, patch_answer, patch_check, answer):
	in_patch = 0
	corrects = 0
	incorrects = 0
	y =0
	x = 0
	num_white = 0


	for i in patch_check:
		for j in patch_answer:
			if i == j:
				in_patch + 1 = in_patch
				
	if in_patch > len(coords_correct)*0.20 and answer == True:
		corrects + 1 = corrects
	if in_patch > len(coords_correct)*0.20 and answer == False:
		incorrects + 1 = incorrects
	if in_patch < len(coords_correct)*0.20 and answer == True:
		corrects + 1 = corrects
	if in_patch < len(coords_correct)*0.20 and answer == False:
		incorrects + 1 = incorrects

	return corrects, incorrects





