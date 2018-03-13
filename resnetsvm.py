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
import pickle


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
    """
    for i in range(len(mass_data)):
        #x = mass_data[i]
        file_name = "C:/Srp 2018/PNGs/mass"+str(i)+".png"
        #dicomToPng(x, file_name)
        img = image.load_img(file_name, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        features = model.predict(x)
        #print(features.shape)
        features_mass.append(features)
    """
    with open("feat_mass.p", "rb") as f:
        features_mass = pickle.load(f)
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

def get_features_convole(mass_data, nonmass_data):
    model = applications.ResNet50(weights='imagenet', include_top=False)
    features_mass = []
    features_nonmass = []
    """                                                                               
    for i in range(len(mass_data)):                                                   
        #x = mass_data[i]                                                             
        file_name = "C:/Srp 2018/PNGs/mass"+str(i)+".png"                             
        #dicomToPng(x, file_name)                                                     
        img = image.load_img(file_name, target_size=(224, 224))                       
        x = image.img_to_array(img)                                                   
        x = np.expand_dims(x, axis=0)                                                 
        features = model.predict(x)                                                   
        #print(features.shape)                                                        
        features_mass.append(features)                                                
    """
    with open("feat_mass.p", "rb") as f:
        features_mass = pickle.load(f)
    for i in range(len(nonmass_data)):
        y = nonmass_data[i]
        stem = "C:/Srp 2018/PNGs/nonmass" + str(i) + "a"
        features_nonmass+=convolve_train(y, stem, model)
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


    clf = sklearn.svm.SVC(probability=True)



    clf.fit(x_svm, y_svm)
    """
    img = image.load_img("C:/Srp 2018/PNGs/nonmass494.png", target_size=(224, 224))
    y = np.expand_dims(image.img_to_array(img)[0:4, 0:4], axis=0)
    topredict = applications.ResNet50(weights='imagenet', include_top=False).predict(y)
    print(clf.predict(np.reshape(topredict, (1, -1))))
    print("made it")
       """
    clf = joblib.dump(clf, "svm_model_convolve_prob.pkl")
def predict_model():
    clf = joblib.load("svm_model.pkl")
    img = image.load_img("C:/Srp 2018/PNGs/nonmass494.png", target_size=(224, 224))
    y = np.expand_dims(image.img_to_array(img)[0:197, 0:197], axis=0)
    topredict = applications.ResNet50(weights='imagenet', include_top=False).predict(y)
    print(topredict.shape)
    print (topredict)
    x_list = []
    x_list.append(topredict.reshape(2048))
    x_svm = np.asarray(x_list)
    print(clf.predict(x_svm))

def convolve_train(img, stem, model):
    """
    y = 0
    x = img.shape[0]-1
    for _ in img.shape[0]*img.shape[1]:
        if img[y, x] != 0:
            break
        if y==img.shape[1]-1:
            y = 0
            x-=1
        y+=1
    img = img[:, 0:x]
    """
    row = 0
    column = 0
    iter = 0
    radius = 56
    features_nonmass = []
    print(img.shape)
    for multirow in range(img.shape[1]//radius):
        for multicolumn in range(img.shape[0]//radius):
            tosave = np.zeros((224, 224))
            tosave[row:row+radius, column:column+radius] = img[row:row+radius, column:column+radius]
            #print(str(row)+" , "+str(column))
            dicomToPng(tosave, stem+str(iter)+".png")
            imago = image.load_img(stem+str(iter)+".png", target_size=(224, 224))
            blah = image.img_to_array(imago)
            blah = np.expand_dims(blah, axis=0)
            features = model.predict(blah)
            #print(features.shape)
            features_nonmass.append(features)
            column+=radius
            iter+=1
        row+=radius
        column = 0
    return features_nonmass
    
def test():
    DDSM = []


def convolve_svm_test(img, stem, model, clf, clf_prob):
    row = 0
    column = 0
    iter = 0
    radius = 56
    print(img.shape)
    for multirow in range(img.shape[1]//radius):
        for multicolumn in range(img.shape[0]//radius):
            tosave = np.zeros((224, 224))
            tosave[row:row+radius, column:column+radius] = img[row:row+radius, column:column+radius]
            dicomToPng(tosave, stem+str(iter)+".png")
            imago = image.load_img(stem+str(iter)+".png", target_size=(224, 224))
            blah = image.img_to_array(imago)
            blah = np.expand_dims(blah, axis=0)
            features = model.predict(blah)
            x_list = []
            x_list.append(features.reshape(2048))
            x_svm = np.asarray(x_list)
            out = clf.predict(x_svm)
            if int(out) == 1:
                init_prob = clf_prob.predict_proba(x_svm)
                init_prob_mass = init_prob[1]
                for new_rad in range(2, radius)[::-6]:
                     newtosave = np.zeros((224, 224))
                     newtosave[(row+radius)/2 - new_rad/2:(row+radius)/2 +new_rad/2, (column+radius)/2 - new_rad/2: (column+radius)/2 + new_rad/2] = img[(row+radius)/2 - new_rad/2:(row+radius)/2 +new_rad/2, (column+radius)/2 - new_rad/2: (column+radius)/2 + new_rad/2]
                     #above line basically has radius go from middle rather than top left
                     #print(str(row)+" , "+str(column))
                     dicomToPng(newtosave, stem+str(iter)+"b"+str(new_rad)+".png")
                     imago = image.load_img(stem+str(iter)+".png", target_size=(224, 224))
                     blah = image.img_to_array(imago)
                     blah = np.expand_dims(blah, axis=0)
                     features = model.predict(blah)
                     x_list = []
                     x_list.append(features.reshape(2048))
                     x_svm = np.asarray(x_list)
                     current_prob_mass = clf_prob.predict_proba(x_svm)[1]
                     if current_prob_mass >= 0.9:
                        print("change in prob for image "+stem+" was "+(current_prob_mass-init_prob_mass))
                        return newtosave
            #print(features.shape)
            column+=radius
            iter+=1
        row+=radius
        column = 0
