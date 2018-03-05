import numpy as np
import pydicom
import median_noise

#uint16 means 0-65,535 grayscale
#returns the best threshold, use for the pectoral_muscle function
def otsu(DDSM):
    threshs = []
    for i in DDSM:
        ds = pydicom.dcmread(i)
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
            f_weight = (len(foreground) / (len(foreground) + len(background)))
            b_weight = (len(background) / (len(foreground) + len(background)))
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
        threshs.append(best_threshold)
    return threshs

def otsu_single(DDSM):
    ds = pydicom.dcmread(DDSM)
    foreground = []
    background = []
    x = 0
    y = 0
    picture = ds.pixel_array
    between_var = []
    threshold_value = []
    for threshold in range(255): #different values for the threshold
        for pixel in range(picture.shape[0]*picture.shape[1]):
            print("in "+str(pixel)+ " pixel and "+str(threshold)+" threshold")
            if picture[y, x] < threshold:
                background.append(picture[y, x])
            if picture[y, x] >= threshold:
                foreground.append(picture[y,x])
            if x == picture.shape[1]:
                y += y
                x = 0
                continue
            x += x
        f_prob = (len(foreground)/(len(foreground)+len(background)))
        b_prob = (len(background)/(len(foreground)+len(background)))
        f_mean = np.mean(foreground)
        b_mean = np.mean(background)
        p_mean = np.mean(picture)
        between_class_variance = b_prob*((b_mean-p_mean)**2) + f_prob*((f_mean-p_mean)**2)
        between_var.append(between_class_variance)
        threshold_value.append(threshold)
    thresh_index = np.argmax(between_var)
    print(thresh_index)
    best_threshold = threshold_value[thresh_index]
    print(best_threshold)
    #set the pixel values
    x = 0
    y = 0
    for i in range(picture.shape[0]*picture.shape[1]):
        if picture[y,x] > best_threshold:
            picture[y, x] = 0
        if x == picture.shape[1]:
            y += y
            x = 0
            continue
        x += x
    ds.PixelData = picture.tostring()
    ds.save_as(DDSM)
    return best_threshold
