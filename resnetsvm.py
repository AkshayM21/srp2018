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

# DDSM.append(get_file.get_file("C:/Srp 2018/Training-Full/Mass-Training_P_00001_LEFT_MLO"))
# SET UP THE DIRECTORY IN THE FUNCTION

"""
DDSM = images
mass_data = list of pixel arrays from mass_v_nonmass for MASS pictures
nonmass_data = ^^ (but for NONMASS pictures)
"""

def getFeatures(DDSM, mass_data, nonmass_data):
    model = applications.ResNet50(weights='imagenet', top_layer=False)
    features_mass = []
    features_nonmass = []
    for i in range(len(DDSM)):
        x = mass_data[i]
        x = np.expand_dims(x, axis=0)
        features = model.predict(x)
        features_mass.append(features)
    for i in range(len(DDSM)):
        y = nonmass_data[i]
        y = np.expand_dims(y, axis = 0)
        features = model.predict(y)
        features_nonmass.append(features)
    return features_mass, features_nonmass


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

    
