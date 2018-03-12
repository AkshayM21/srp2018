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

    for i in range(len(DDSM_ROI))[::-1]:
        try:
            ddsmDICOM = pydicom.dcmread(ddsm_roi.getDDSMequivalent(DDSM_ROI[i]))
            ddsmroiDICOM = pydicom.dcmread(DDSM_ROI[i])
           # print(ddsm_roi.getDDSMequivalent(DDSM_ROI[i])+" x "+DDSM_ROI[i])
            try:

                ROI_Array = np.multiply(ddsmDICOM.pixel_array, ddsm_roi.make_mask(ddsmroiDICOM.pixel_array))
            except AttributeError:
                print("uh oh ROI split at "+DDSM_ROI[i])
                DDSM_ROI.remove(DDSM_ROI[i])
                continue
        except IndexError:
            blahblah = 1
            continue
        mass.append(ROI_Array)

    return mass


#CHANGE PIXEL TO BE PART OF THE ARRAY AND CREAT AN INVERTED ARRAY
def ROI_Inverse(DDSM_ROI):

    roi_split = []

    for i in range(len(DDSM_ROI))[::-1]:
        x = 0
        y = 0
        try:
            ddsmroiDICOM = pydicom.dcmread(DDSM_ROI[i])
            try:
                ddsmroi_array = ddsmroiDICOM.pixel_array.copy()
            except AttributeError:
                print("uh oh roi_inverse at "+DDSM_ROI[i])
                DDSM_ROI.remove(DDSM_ROI[i])
                continue
        except IndexError:
            blahblah = 1
            continue
        y = 0
        x = 0
        for pixel in range(ddsmroi_array.shape[0]*ddsmroi_array.shape[1]):
            try:
                if ddsmroi_array[y, x] == 65535:
                    ddsmroi_array[y, x] = 0
                if ddsmroi_array[y,x] == 0:
                    ddsmroi_array[y,x] = 1
                if x == ddsmroi_array.shape[0] - 1:
                    y = y + 1
                    x = 0
                    continue
                x += 1
            except IndexError:
                continue
        roi_split.append(ddsmroi_array)
    return roi_split


def DDSM_Split(DDSM, roi_split, DDSMROI):
    import ddsm_roi
    non_mass = []
    for i in range(len(DDSMROI))[::-1]:
        try:
            ddsmDICOM = pydicom.dcmread(ddsm_roi.getDDSMequivalent(DDSMROI[i]))
            try:
                Invert_Array = np.multiply(ddsmDICOM.pixel_array, roi_split[i])
            except AttributeError:
                print("uh oh at DDSM_split at "+DDSMROI[i])
                DDSMROI.remove(DDSMROI[i])
                roi_split.remove(roi_split[i])
                continue
            except ValueError:
                continue
                if roi_split[i].shape[0]!=224 or roi_split[i].shape[1]!=224:
                    #crop
                    Rows = roi_split[i].shape[1]
                    Columns = roi_split[i].shape[0]
                    if (Rows > Columns):
                        crop_arr = roi_split[i][(Rows / 2) - (Columns / 2):(Rows / 2) + (Columns / 2), :]
                        Rows = Columns
                    else:
                        crop_arr = roi_split[i][:, (Columns / 2) - (Rows / 2):(Columns / 2) + (Rows / 2)]
                        Columns = Rows
                    import cv2
                    roi_split[i] = cv2.resize(crop_arr, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
                    Invert_Array = np.multiply(ddsmDICOM.pixel_array, roi_split[i])
        except IndexError:
            blahblah = 1
            continue
        non_mass.append(Invert_Array)
    return non_mass