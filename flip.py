import numpy as np
import dicom

#ddsm is a list of file names
def flip(DDSM):
    for i in (DDSM):
        ds = dicom.read_file(i);
        str_len = len(i);
        



