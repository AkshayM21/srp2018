import dicom
import numpy as np


"""
Steps:
1. Estimate threshold (Th0)
2. Apply threshold (Th0) to original image (I) to get binary image (B0)
3. Apply connected component algorithm to binary image (B0) to
identify separate objects
4. Estimate the area (A0) of the upper-left object
5. Apply binary image (B0) to original image (I) to get gray scale
image (I0)
6. Ak-1 = A0, Ik-1 = I0
7. for (k = 1 to 5)
8. Estimate threshold (Thk)
9. Apply threshold (Thk) to (Ik-1) to get binary image (Bk)
10. Apply connected component algorithm to binary image (Bk) to
identify separate objects
11. Estimate the area (Ak) of the upper-right object
12. Apply binary image (Bk) to (Ik-1) to get (Ik)
13. if (Ak = Ak-1)
14. break
15. else
16. Ik-1 = Ik, Ak-1 = Ak
17. Generate binary image (Bf) from (Ik) with
18. Bf = 0, for the upper-left object
19. Bf = 1, otherwise
20. Apply binary image (Bf) to get the final image (If) without the
pectoral muscle
"""

#get init thresh from otsu's method
def remove_pec(filename, init_thresh):
    thresh = init_thresh
    ds = dicom.read_file(filename)
    bin_img = apply_mask(thresh, ds)
    pec_muscle = connected_comp(bin_img, False, True)
    for k in range(1, 5):


#placeholder - remember to import the real one from andy
def apply_mask(threshold, dataset):



#placeholder - remember to import the real one
def connected_comp(image, left, top):


