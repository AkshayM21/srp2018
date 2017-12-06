import numpy as np
import dicom
from algorithms.sorting import quick_sort


for i in (DDSM):
	ds = dicom.read_file(i)
	shape = ds.pixel_array.shape
	# initialize the y part of the pixel in the array
	y = 0
	#initialize x
	x = 0

	#size of image
	pixels = shape[0]*shape[1]
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
			y + 1 = y
			x = 0
	ds.PixelData = ds.pixel_array.tostring()
	ds.save_as(i)



