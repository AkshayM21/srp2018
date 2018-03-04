import numpy as np
import pydicom
from algorithms.sorting import quick_sort

def noise_removal(DDSM):
    for i in (DDSM):
        ds = pydicom.dcmread(i)
        #initialize the y part of the pixel in the array
        y = 0
        #initialize x
        x = 0
        #size of image
        pixels = ds.pixel_array.shape[0]*ds.pixel_array.shape[1]
        for pixel in range(pixels):
            # define the pixel we're looking at
            n = ds.pixel_array[y , x]
            n_up = ds.pixel_array[y - 1, x]
            n_upl = ds.pixel_array[y - 1, x - 1]
            n_l = ds.pixel_array[y, x - 1]
            n_downl = ds.pixel_array[y + 1, x - 1]
            n_down = ds.pixel_array[y + 1, x]
            n_downr = ds.pixel_array[y + 1, x + 1]
            n_r = ds.pixel_array[y, x + 1]
            n_upr = ds.pixel_array[y - 1, x + 1]
            window = [n, n_up, n_upr, n_upl, n_l, n_downl, n_down, n_downr, n_r]
            #sorting
            window = sorted(window)
            #set value to pixel
            ds.pixel_array[y, x] = window[4]
            if x == ds.pixel_array.shape[1]:
                y = y + 1
                x = 0
        ds.PixelData = ds.pixel_array.tostring()
        ds.save_as(i)

def noise_removal_single(i):
    ds = pydicom.dcmread(i)
    #initialize the y part of the pixel in the array
    y = 0
    #initialize x
    x = 0
    #size of image
    pixels = ds.pixel_array.shape[0]*ds.pixel_array.shape[1]
    for pixel in range(pixels):
        # define the pixel we're looking at
        n = ds.pixel_array[y , x]
        n_up = ds.pixel_array[y - 1, x]
        n_upl = ds.pixel_array[y - 1, x - 1]
        n_l = ds.pixel_array[y, x - 1]
        n_downl = ds.pixel_array[y + 1, x - 1]
        n_down = ds.pixel_array[y + 1, x]
        n_downr = ds.pixel_array[y + 1, x + 1]
        n_r = ds.pixel_array[y, x + 1]
        n_upr = ds.pixel_array[y - 1, x + 1]
        window = [n, n_up, n_upr, n_upl, n_l, n_downl, n_down, n_downr, n_r]
        #sorting
        window = quick_sort.sort(window)
        #set value to pixel
        ds.pixel_array[y, x] = window[4]
        if x == ds.pixel_array.shape[1]:
            y = y + 1
            x = 0
        print("done with iteration " + str(pixel) +" for median noise")
    ds.PixelData = ds.pixel_array.tostring()
    ds.save_as(i)