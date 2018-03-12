import pydicom
import preprocessing
import resnetsvm
import pickle
import os

mass, nonmass = preprocessing.final_preprocess()
print("done w/ preprocess")
with open("nonmass.p", "wb") as f:
    pickle.dump(nonmass, f)
with open("mass.p", "wb") as f:
    pickle.dump(mass, f)
#with open("nonmass.p", "rb") as f:
#    nonmass = pickle.load(f)
#with open("mass.p", "rb") as f:
#    mass = pickle.load(f)
features_mass, features_nonmass = resnetsvm.getFeatures(mass, nonmass)
print("done w/ resnet")
resnetsvm.svm(features_nonmass, features_mass)
print("completely done")





