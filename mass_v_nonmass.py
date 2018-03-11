import pydicom

mass = []
non_mass = []

def ROI_Split(DDSM, DDSM_ROI):
	for i in DDSM:

		mass.append(DDSM * DDSM_ROI)

#CHANGE PIXEL TO BE PART OF THE ARRAY AND CREAT AN INVERTED ARRAY
def ROI_Inverse(DDSM_ROI):
	for i in DDSM_ROI:
		for pixel in i:
			if pixel = 1:
				pixel = 0


def DDSM_Split(DDSM):
	for i in DDSM:

		nonmass.append(DDSM*ROI_Invert)
