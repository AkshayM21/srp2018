import numpy as np
import dicom

#ddsm is a list of file names
def flip(DDSM):
    for i in (DDSM):
        ds = dicom.read_file(i)
        list = split(i, "_",)
        if list[len(list)-1] == "1":
            if list[len(list)-3] == "LEFT":
                height = ds.pixel_array.shape[0]
                width = ds.pixel_array.shape[1]
                for x in range(0, width / 2):  # Only process the half way
                    for y in range(0, height):
                        # swap pix and pix2
                        one = ds.pixel_array[y, x]
                        two = ds.pixel_array[y, width-1-x]
                        ds.pixel_array[y, x] = two
                        ds.pixel_array[y, width-1-x] = one
                ds.PixelData = ds.pixel_array.tostring()
        else:
            if list[len(list)-2] == "LEFT":
                height = ds.pixel_array.shape[0]
                width = ds.pixel_array.shape[1]
                for x in range(0, width / 2):  # Only process the half way
                    for y in range(0, height):
                        # swap pix and pix2
                        one = ds.pixel_array[y, x]
                        two = ds.pixel_array[y, width - 1 - x]
                        ds.pixel_array[y, x] = two
                        ds.pixel_array[y, width - 1 - x] = one
                ds.PixelData = ds.pixel_array.tostring()





