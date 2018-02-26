import numpy as np
import dicom
import median_noise

#uint16 means 0-65,535 grayscale
#returns the best threshold, use for the pectoral_muscle function
def otsu(DDSM):
	for i in DDSM:
		ds = dicom.read_file(i.tostring())
		foreground = []
		background = []
		x = 0
		y = 0
		picture = ds.pixel_array
		threshold_value = []
		for threshold in range(65536): #different values for the threshold
			for pixel in range(picture.shape[0]*picture.shape[1]):
				
				if picture[y, x] < threshold:
					background.append(picture[y, x])
				if picture[y, x] >= threshold:
					foreground.append(picture[y,x])

				if x == picture.shape[1]:
					y += y
					x = 0
					continue
				x += x
			f_weight = (foreground.length()/(foreground.length()+background.length()))
			b_weight = (background.length()/(foreground.length()+background.length()))
			f_mean = np.mean(foreground)
			b_mean = np.mean(background)
			f_variance = np.var(foreground) #check this function
			b_variance = np.var(background)

			within_class_variance = f_weight*f_variance + b_weight*b_variance
			threshold_value.append(within_class_variance)
		best_threshold = max(threshold_value)
		#set the pixel values
		x = 0
		y = 0
		for i in range(picture.shape[0]*picture.shape[1]):
			if picture[y,x] < best_threshold:
				picture[y, x] = 0
			if picture[y,x] >= best_threshold:
				picture[y, x] = 1

			if x == picture.shape[1]: 
				y += y
				x = 0
				continue
			x += x
		ds.PixelData = picture.tostring()
		ds.save_as(i)
		return best_threshold






	