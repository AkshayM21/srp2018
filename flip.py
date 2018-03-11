import numpy as np
import pydicom
import cv2

#ddsm is a list of file names
def flip(DDSM):
    for i in DDSM:
        ds = pydicom.dcmread(i)
        list = i.split("_")
        if list[len(list) - 1] == "1" or list[len(list) - 1] == "2":
            if list[len(list) - 3] == "RIGHT":
                """
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
                ds.save_as(i)
                """
                pixel_arr = cv2.flip(ds.pixel_array, 1)
                ds.PixelData = pixel_arr.tostring()
                ds.save_as(i)
        else:
            if list[len(list) - 2] == "RIGHT":
                """
                height = ds.pixel_array.shape[0]
                width = ds.pixel_array.shape[1]
                for x in range(0, width / 2):  # Only process the half way
                    for y in range(0, height):
                        # swap pix and pix2
                        one = ds.pixel_array[y, x]
                        two = ds.pixel_array[y, width - 1 - x]
                        ds.pixel_array[y, x] = two
                        ds.pixel_array[y, width - 1 - x] = one
                        """
                pixel_arr = cv2.flip(ds.pixel_array, 1)
                ds.PixelData = pixel_arr.tostring()
                ds.save_as(i)



def flip_single(i):
    ds = pydicom.dcmread(i)
    list = i.split("_")
    if list[len(list)-1] == "1" or list[len(list)-1] == "2":
        if list[len(list)-3] == "RIGHT":
            """
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
            ds.save_as(i)
            """
            pixel_arr = cv2.flip(ds.pixel_array, 1)
            ds.PixelData = pixel_arr.tostring()
            ds.save_as(i)
    else:
        if list[len(list)-2] == "RIGHT":
            """
            height = ds.pixel_array.shape[0]
            width = ds.pixel_array.shape[1]
            for x in range(0, width / 2):  # Only process the half way
                for y in range(0, height):
                    # swap pix and pix2
                    one = ds.pixel_array[y, x]
                    two = ds.pixel_array[y, width - 1 - x]
                    ds.pixel_array[y, x] = two
                    ds.pixel_array[y, width - 1 - x] = one
                    """
            pixel_arr = cv2.flip(ds.pixel_array, 1)
            ds.PixelData = pixel_arr.tostring()
            ds.save_as(i)