import tensorflow
from keras import applications
import keras
import numpy as np
import sklearn
import pydicom
import get_file

model = applications.ResNet50(weights='imagenet', top_layer = False)

independent = []
DDSM = []
DDSM.append(get_file.get_file("C:/Srp 2018/Training-Full/Mass-Training_P_00001_LEFT_MLO"))
for i in DDSM:
	img_path = i
	x = pydicom.dcmread(img_path).pixel_array
	x = np.expand_dims(x, axis=0)
	features = model.predict(x)
	independent.append(features)


"""
clf = sklearn.svm.SVC()

x,y = independent, DDSM_ROI

clf.fit(x, y)
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

"""