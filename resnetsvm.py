import tensorflow
from keras import applications
import keras
import numpy as np
import sklearn
from sklearn.svm import SVC
import pydicom
import get_file
import mass_v_nonmass
from sklearn.externals import joblib
import h5py
from keras.applications.resnet50 import preprocess_input, decode_predictions
import png
from keras.preprocessing import image


# DDSM.append(get_file.get_file("C:/Srp 2018/Training-Full/Mass-Training_P_00001_LEFT_MLO"))
# SET UP THE DIRECTORY IN THE FUNCTION

"""
DDSM = images
mass_data = list of pixel arrays from mass_v_nonmass for MASS pictures
nonmass_data = ^^ (but for NONMASS pictures)
"""

def getFeatures(length, mass_data, nonmass_data):
    model = applications.ResNet50(weights='imagenet', include_top=False)
    features_mass = []
    features_nonmass = []
    for i in range(length):
        x = mass_data[i]
        file_name = "C:/Srp 2018/PNGs/mass"+str(i)+".png"
        dicomToPng(x, file_name)
        img = image.load_img(file_name, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        features = model.predict(x)
        features_mass.append(features)
    for i in range(length):
        y = nonmass_data[i]
        file_name = "C:/Srp 2018/PNGs/nonmass" + str(i) + ".png"
        dicomToPng(y, file_name)
        img = image.load_img(file_name, target_size=(224, 224))
        y = image.img_to_array(img)
        y = np.expand_dims(y, axis=0)
        features = model.predict(y)
        features_nonmass.append(features)
    return features_mass, features_nonmass

def dicomToPng(pixel_array, file_name):

    shape = pixel_array.shape

    png_file = open(file_name, 'wb')

    image_2d = []
    max_val = 0
    for row in pixel_array:
        pixels = []
        for col in row:
            pixels.append(col)
            if col > max_val: max_val = col
        image_2d.append(pixels)

    # Rescaling grey scale between 0-255
    image_2d_scaled = []
    for row in image_2d:
        row_scaled = []
        for col in row:
            col_scaled = int((float(col) / float(max_val)) * 255.0)
            row_scaled.append(col_scaled)
        image_2d_scaled.append(row_scaled)

    # Writing the PNG file
    w = png.Writer(shape[0], shape[1], greyscale=True)
    w.write(png_file, image_2d_scaled)

"""
labels:
1 === mass
0 === nonmass
"""
def svm(features_nonmass, features_mass):
    non_mass_labels = np.zeros((len(features_nonmass)))
    mass_labels = np.full((len(features_mass)), 1)
    x_svm = []
    y_svm = []

    x_svm.append(features_mass)
    x_svm.append(features_nonmass)
    y_svm.append(mass_labels)
    y_svm.append(non_mass_labels)

    clf = sklearn.svm.SVC()


    clf.fit(x_svm, y_svm)

    clf = joblib.dump(clf, "svm_model.pkl")

