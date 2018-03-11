import pydicom
import preprocessing
import resnetsvm

mass, nonmass, DDSM = preprocessing.final_preprocess()
features_mass, features_nonmass = resnetsvm.getFeatures(DDSM, mass, nonmass)
resnetsvm.svm(features_nonmass, features_mass)

