import tensorflow
from keras import applications
import keras
import numpy as np
import sklearn

model = applications.ResNet50(weights='imagenet', top_layer = False)

independent = []

for i in DDSM:
	img_path = DDSM
	img = image.load_img(img_path, target_size=(224, 224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)

	features = model.predict(x)
	independent.append(features)



clf = svm.SVC()

x,y = independent, DDSM_AOI

clf.fit(X, y)  
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)