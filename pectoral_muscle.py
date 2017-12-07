import dicom
import numpy as np

def remove_pec(filename):
    ds = dicom.read_file(filename)

#get initial threshold using lloyd-max algorithm
def init_thresh():



