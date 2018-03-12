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

def getFeatures(mass_data, nonmass_data):
    model = applications.ResNet50(weights='imagenet', include_top=False)
    features_mass = []
    features_nonmass = []
    for i in range(len(mass_data)):
        x = mass_data[i]
        file_name = "C:/Srp 2018/PNGs/mass"+str(i)+".png"
        dicomToPng(x, file_name)
        img = image.load_img(file_name, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        features = model.predict(x)
        #print(features.shape)
        features_mass.append(features)
    for i in range(len(nonmass_data)):
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

    # Writing the PNG file
    w = png.Writer(shape[0], shape[1], greyscale=True)
    w.write(png_file, np.divide(pixel_array, 256))
"""
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
        """





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

    x_list = []
    for i in range(len(features_mass)):
        x_list.append(np.reshape(features_mass[i], 2048))

    for j in range(len(features_nonmass)):
        x_list.append(np.reshape(features_nonmass[j], 2048))

    #x_svm = np.concatenate((features_mass, features_nonmass), axis=0)

    y_svm = np.append(mass_labels, non_mass_labels)

    x_svm = np.asarray(x_list)


    clf = sklearn.svm.SVC()



    clf.fit(x_svm, y_svm)

    img = image.load_img("C:/Srp 2018/PNGs/mass1.png", target_size=(224, 224))
    y = np.expand_dims(image.img_to_array(img)[0:197, 0:197], axis=0)
    topredict = applications.ResNet50(weights='imagenet', include_top=False).predict(y)
    print(clf.predict(np.reshape(topredict, (1, -1))))
    print("made it")

    #clf = joblib.dump(clf, "svm_model.pkl")

