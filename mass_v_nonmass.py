import pydicom
import pectoral_muscle
import numpy as np
import os


mass = []
roi_split = []
non_mass = []


def ROI_Split(DDSM, DDSM_ROI):
    mass = []

    import ddsm_roi

    for i in range(len(DDSM_ROI)):
        ddsmDICOM = pydicom.dcmread(ddsm_roi.getDDSMequivalent(DDSM_ROI[i]))
        ddsmroiDICOM = pydicom.dcmread(DDSM_ROI[i])
        ROI_Array = np.multiply(ddsmDICOM.pixel_array, ddsm_roi.make_mask(ddsmroiDICOM.pixel_array))
        mass.append(ROI_Array)

    return mass




#CHANGE PIXEL TO BE PART OF THE ARRAY AND CREAT AN INVERTED ARRAY
def ROI_Inverse(DDSM_ROI):

    roi_split = []

    for i in range(len(DDSM_ROI)):
        x = 0
        y = 0
        ddsmroiDICOM = pydicom.dcmread(DDSM_ROI[i])
        ddsmroi_array = ddsmroiDICOM.pixel_array.copy()
        for pixel in range(ddsmroi_array.shape[0]*ddsmroi_array.shape[1]):
            if ddsmroi_array[y, x] == 65535:
                ddsmroi_array[y, x] = 0
            if ddsmroi_array[y,x] == 0:
                ddsmroi_array[y,x] = 1
            if x == ddsmroi_array.shape[0] - 1:
                y = y + 1
                x = 0
                continue
            x += 1
        roi_split.append(ddsmroi_array)
    return roi_split



def DDSM_Split(DDSM, roi_split, DDSMROI):
    import ddsm_roi
    non_mass = []
    for i in range(len(DDSMROI)):

        ddsmDICOM = pydicom.dcmread(ddsm_roi.getDDSMequivalent(DDSMROI[i]))

        Invert_Array = np.multiply(ddsmDICOM.pixel_array, roi_split[i])
        non_mass.append(Invert_Array)
    return non_mass