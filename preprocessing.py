import median_noise
import get_file
import flip
import artifact_removal
import pectoral_muscle
import cv2
import pydicom
import replace
from skimage import filters
import ddsm_roi
import mass_v_nonmass


"""
init_folder = "D:/Akshay SRP 2018/Test-Full/CBIS-DDSM/"
DDSM = get_file.get_full_path(init_folder)
print(DDSM)
flip.flip(DDSM)
print("done w/ flip")
for i in DDSM:
    ds = pydicom.dcmread(i)
    nparr = ds.pixel_array
    if(ds.Rows>ds.Columns):
        crop_arr = nparr[(ds.Rows/2)-(ds.Columns/2):(ds.Rows/2)+(ds.Columns/2), :]
        ds.Rows = ds.Columns
    else:
        crop_arr = nparr[:, (ds.Columns/2)-(ds.Rows/2):(ds.Columns/2)+(ds.Rows/2)]
        ds.Columns = ds.Rows
    res = cv2.resize(crop_arr, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
    ds.Rows = 224
    ds.Columns = 224
    ds.PixelData = res.tostring()
    ds.save_as(i)
print("done with crop")
median_noise.noise_removal(DDSM)
print("done with noise removal")
for i in DDSM:
    thresh = filters.threshold_otsu(pydicom.dcmread(i).pixel_array)
    pectoral_muscle.canny_remove(i)
print("done w/ otsu's and pectoral muscle")
print("done")
"""

init_folder = "D:/Akshay SRP 2018/Test-ROI/CBIS-DDSM/"
DDSMroi = ddsm_roi.get_roi(init_folder)
flip.flip(DDSMroi)
print("done w/ flip roi")
for i in DDSMroi:
    ds = pydicom.dcmread(i)
    try:
        nparr = ds.pixel_array
    except AttributeError:
        print("uh oh pass through at "+i)
        DDSMroi.remove(i)
        continue
    if(ds.Rows>ds.Columns):
        crop_arr = nparr[(ds.Rows/2)-(ds.Columns/2):(ds.Rows/2)+(ds.Columns/2), :]
        ds.Rows = ds.Columns
    else:
        crop_arr = nparr[:, (ds.Columns/2)-(ds.Rows/2):(ds.Columns/2)+(ds.Rows/2)]
        ds.Columns = ds.Rows
    res = cv2.resize(crop_arr, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
    ds.Rows = 224
    ds.Columns = 224
    ds.PixelData = res.tostring()
    ds.save_as(i)
print("done with crop roi")
print("done roi")

def final_preprocess():
    init_folder1 = "C:/Srp 2018/Training-Full/"
    DDSM = get_file.get_full_path(init_folder1)
    #print(DDSM)
    #print(len(DDSM))
    init_folder = "C:/Srp 2018/Training-ROI/CBIS-DDSM/"
    ROI, DDSM = ddsm_roi.get_roi_cropped(init_folder, DDSM)
    mass = mass_v_nonmass.ROI_Split(DDSM, ROI)
    roi_split = mass_v_nonmass.ROI_Inverse(ROI)
    non_mass =  mass_v_nonmass.DDSM_Split(DDSM, roi_split, ROI)

    return mass, non_mass




def preprocess_main():
    init_folder = "D:/Akshay SRP 2018/Training-Full/"
    DDSM = get_file.get_full_path(init_folder)
    print(DDSM)
    flip.flip(DDSM)
    print("done w/ flip")
    for i in DDSM:
        ds = pydicom.dcmread(i)
        nparr = ds.pixel_array
        if(ds.Rows>ds.Columns):
            crop_arr = nparr[(ds.Rows/2)-(ds.Columns/2):(ds.Rows/2)+(ds.Columns/2), :]
            ds.Rows = ds.Columns
        else:
            crop_arr = nparr[:, (ds.Columns/2)-(ds.Rows/2):(ds.Columns/2)+(ds.Rows/2)]
            ds.Columns = ds.Rows
        res = cv2.resize(crop_arr, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
        ds.Rows = 224
        ds.Columns = 224
        ds.PixelData = res.tostring()
        ds.save_as(i)
    print("done with crop")
    median_noise.noise_removal(DDSM)
    print("done with noise removal")
    for i in DDSM:
        thresh = filters.threshold_otsu(pydicom.dcmread(i).pixel_array)
        pectoral_muscle.canny_remove(i)
    print("done w/ otsu's and pectoral muscle")
    print("done")

"""

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