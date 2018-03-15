import pydicom
import preprocessing
import resnetsvm
import pickle
import os

#mass, nonmass = preprocessing.final_preprocess_vgg()
#print("done w/ preprocess")
#with open("nonmass.p", "wb") as f:
#    pickle.dump(nonmass, f)
#with open("mass.p", "wb") as f:
#    pickle.dump(mass, f)
##with open("nonmass.p", "rb") as f:
##    nonmass = pickle.load(f)
##with open("mass.p", "rb") as f:
##    mass = pickle.load(f)
#features_mass, features_nonmass = resnetsvm.get_features_vgg(mass, nonmass)
#print("done w/ vgg")
#with open("feat_nonmass.p", "wb") as f:
#    pickle.dump(features_nonmass, f)
#with open("feat_mass.p", "wb") as f:
#    pickle.dump(features_mass, f)
#with open("feat_nonmass.p", "rb") as f:
#   features_nonmass = pickle.load(f)
#with open("feat_mass.p", "rb") as f:
#    features_mass = pickle.load(f)
###resnetsvm.logistic_reg(features_nonmass, features_mass)
#resnetsvm.svm_vgg(features_nonmass, features_mass)
resnetsvm.test()
print("completely done")
#resnetsvm.predict_model()





