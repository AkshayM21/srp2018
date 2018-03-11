import pydicom
import pectoral_muscle
import numpy
import os


mass = []
roi_split = []
non_mass = []


def ROI_Split(DDSM, DDSM_ROI):
    mass = []

    for i in range(len(DDSM)):
        ddsmDICOM = pydicom.dcmread(DDSM[i])
        ddsmroiDICOM = pydicom.dcmread(DDSM_ROI[i])
        ROI_Array = ddsmDICOM.pixel_array * ddsmroiDICOM.pixel_array
        mass.append(ROI_Array)

    return mass




#CHANGE PIXEL TO BE PART OF THE ARRAY AND CREAT AN INVERTED ARRAY
def ROI_Inverse(DDSM_ROI):

    roi_split = []

    for i in range(len(DDSM_ROI)):
        ddsmroiDICOM = pydicom.dcmread(DDSM_ROI[i])
        for pixel in ddsmroiDICOM.pixel_array:
            if pixel == 1:
                pixel = 0
            if pixel == 0:
                pixel = 1
            if x == ddsmroiDICOM.pixel_array.shape[0] - 1:
                y = y + 1
                x = 0
                continue
            x += 1
        roi_split.append(ddsmroiDICOM.pixel_array)
    return roi_split



def DDSM_Split(DDSM, roi_split):
    non_mass = []
    for i in range(len(DDSM)):

        ddsmDICOM = pydicom.dcmread(DDSM[i])

        Invert_Array = ddsmDICOM.pixel_array * roi_split[i]
        non_mass.append(Invert_Array)
    return non_mass