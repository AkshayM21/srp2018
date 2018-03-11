import pydicom
import pectoral_muscle
import numpy
import os


mass = []
roi_split = []
non_mass = []


def ROI_Split(DDSM, DDSM_ROI):


	for i in range(len(DDSM)):
		
        ddsmDICOM = pydicom.dcmread(DDSM[i])
        ddsmroiDICOM = pydicom.dcmread(DDSM_ROI[i])

		ROI_Array = ddsmDICOM.pixel_array * ddsmroiDICOM.pixel_array
		mass.append(ROI_Array)



#CHANGE PIXEL TO BE PART OF THE ARRAY AND CREAT AN INVERTED ARRAY
def ROI_Inverse(DDSM_ROI):


	
	for i in range(len(DDSM_ROI)):
        ddsmroiDICOM = pydicom.dcmread(DDSM_ROI[i])

		for pixel in ddsmroiDICOM.pixel_array:
			if pixel = 1:
				pixel = 0
			if pixel = 0:
				pixel = 1
			if x == ds.pixel_array.shape[1] - 1:
                y = y + 1
                x = 0
                continue
            x += 1
        roi_split.append(ddsmroiDICOM.pixelarray)



def DDSM_Split(DDSM):
	for i in range(len(DDSM)):
		
        ddsmDICOM = pydicom.dcmread(DDSM[i])

		Invert_Array = ddsmDICOM.pixel_array * roi_split[i]
		non_mass.append(Invert_Array)
