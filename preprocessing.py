import median_noise
import get_file
import flip
import artifact_removal
import pectoral_muscle
import cv2
import pydicom

import sys
import threading

DDSM = get_file.get_file("C:\Srp 2018\Mass-Training_P_00001_LEFT_MLO")+"/000000.dcm"
print(DDSM)
flip.flip_single(DDSM)
print("after flip")
#resizing the image to 224 x 224 for the model
ds = pydicom.dcmread(DDSM)
nparr = ds.pixel_array
if(ds.Rows>ds.Columns):
    crop_arr = nparr[(ds.Rows/2)-(ds.Columns/2):(ds.Rows/2)+(ds.Columns/2), :]
    ds.Rows = ds.Columns
else:
    crop_arr = nparr[:, (ds.Columns/2)-(ds.Rows/2):(ds.Columns/2)+(ds.Rows/2)]
    ds.Columns = ds.Rows
print(ds)
res = cv2.resize(crop_arr, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
ds.Rows = 224
ds.Columns = 224
ds.PixelData = res.tostring()
ds.save_as(DDSM)
print("after resize")
median_noise.noise_removal_single(DDSM)
print("after noise removal")
thresh = artifact_removal.otsu_single(DDSM)
print(thresh)
print("after artifact removal")
pectoral_muscle.remove_pec(DDSM, thresh)
print("done w/ preprocess")

"""
steps needed:
1. noise reduction
2. img flip
3. otsu's
4. pectoral muscle removal

todo:
integrate four above in one continuous thing
try it out on one image (copied) - right and left
if that works fully then do it for all


#goes through preprocessing pipeline
def preprocess(init_folder):
    DDSM = get_file.get_full_path(init_folder)
    median_noise.noise_removal(DDSM)
    flip.flip(DDSM)
    threshs = artifact_removal.otsu(DDSM)
    for i in range(len(DDSM)):
        pectoral_muscle.remove_pec(DDSM[i], threshs[i])
        #resizing the image to 224 x 224 for the model
        ds = dicom.read_file(DDSM[i])
        nparr = ds.pixel_array
        res = cv2.resize(nparr, dsize=(224,224), interpolation=cv2.INTER_CUBIC)
        ds.PixelData = res.tostring()
        ds.save_as(i)
"""