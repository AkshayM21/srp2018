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
import correct_test


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
        #x = mass_data[i]
        file_name = "C:/Srp 2018/PNGs/mass"+str(i)+".png"
        #dicomToPng(x, file_name)
        img = image.load_img(file_name, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        features = model.predict(x)
        #print(features.shape)
        features_mass.append(features)
    with open("feat_mass.p", "rb") as f:
        features_mass = pickle.load(f)
    for i in range(len(nonmass_data)):
        print(i)
        y = nonmass_data[i]
        file_name = "C:/Srp 2018/PNGs/nonmass" + str(i) + ".png"
        dicomToPng(y, file_name)
        img = image.load_img(file_name, target_size=(224, 224))
        y = image.img_to_array(img)
        y = np.expand_dims(y, axis=0)
        features = model.predict(y)
        features_nonmass.append(features)
    return features_mass, features_nonmass

def get_features_vgg(mass_data, nonmass_data):
    model = applications.VGG19(weights='imagenet', include_top=False, input_shape=(56, 56, 3))
    features_mass = []
    features_nonmass = []
    for i in range(len(mass_data)):
        x = mass_data[i]
        file_name = "C:/Srp 2018/PNGs/mass" + str(i) + ".png"
        dicomToPng(x, file_name)
        img = image.load_img(file_name, target_size=(56, 56))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        features = model.predict(x)
        # print(features.shape)
        features_mass.append(features)
    with open("feat_mass.p", "rb") as f:
        features_mass = pickle.load(f)
    for i in range(len(nonmass_data)):
        print(i)
        y = nonmass_data[i]
        stem = "C:/Srp 2018/PNGs/nonmass" + str(i) + "a"
        features_nonmass += convolve_train_vgg(y, stem, model)
    return features_mass, features_nonmass



def get_features_convolve(mass_data, nonmass_data):
    model = applications.ResNet50(weights='imagenet', include_top=False, input_shape=(56, 56, 3))
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
    with open("feat_mass.p", "rb") as f:
        features_mass = pickle.load(f)
    iter = len(mass_data)
    for i in range(len(nonmass_data)):
        print(i)
        y = nonmass_data[i]
        stem = "C:/Srp 2018/PNGs/nonmass" + str(i) + "a"
        feat_nonmass, iter = convolve_train_noblack(y, stem, model, iter)
        if iter<=0: break
        features_nonmass+=   feat_nonmass
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

def logistic_reg(features_nonmass, features_mass):
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

    clf = sklearn.linear_model.LogisticRegression()
    

    clf.fit(x_svm, y_svm)
    clf = joblib.dump(clf, "log_regress.pkl")


def svm(features_nonmass, features_mass):
    non_mass_labels = np.zeros((len(features_nonmass)))
    mass_labels = np.full((len(features_mass)), 1)
    x_svm = []
    y_svm = []

    x_list = []
    for i in range(len(features_nonmass)):
        x_list.append(np.reshape(features_nonmass[i], 2048))

    for j in range(len(features_mass)):
        x_list.append(np.reshape(features_mass[j], 2048))

    #x_svm = np.concatenate((features_mass, features_nonmass), axis=0)

    y_svm = np.append(non_mass_labels, mass_labels)

    x_svm = np.asarray(x_list)

    clf = sklearn.svm.SVC()
    clf_prob = sklearn.svm.SVC(probability=True)

    clf.fit(x_svm, y_svm)
    clf_prob.fit(x_svm, y_svm)
    """
    img = image.load_img("C:/Srp 2018/PNGs/nonmass494.png", target_size=(224, 224))
    y = np.expand_dims(image.img_to_array(img)[0:4, 0:4], axis=0)
    topredict = applications.ResNet50(weights='imagenet', include_top=False).predict(y)
    print(clf.predict(np.reshape(topredict, (1, -1))))
    print("made it")
       """
    clf = joblib.dump(clf, "svm_vgg.pkl")
    clf_prob = joblib.dump(clf_prob, "svm_vgg_prob.pkl")


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

def convolve_train_noblack(img, stem, model, len_mass):
    row = 0
    column = 0
    iter = 0
    radius = 50
    features_nonmass = []
    #print(img.shape)
    for multirow in range(img.shape[1]//radius):
        for multicolumn in range(img.shape[0]//radius):
            if window_percent_zeros(img[row:row+radius, column:column+radius])<=0.3:
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
                len_mass-=1
                if len_mass ==0: break
        row+=radius
        column = 0
    return features_nonmass, len_mass

def window_percent_zeros(pixel_array):
    y = 0
    x = -1
    num_zeros = 0
    for i in range(pixel_array.shape[0]*pixel_array.shape[1]):
        if x == pixel_array.shape[0]-1:
            x = 0
            y  = y+ 1
        else:
            x = x + 1
        if pixel_array[y, x] == 0:
            num_zeros+=1
    return num_zeros/(pixel_array.shape[0]*pixel_array.shape[1])

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
    import ddsm_roi
    DDSM = get_file.get_full_path("C:/Srp 2018/Test-Full/CBIS-DDSM/")
    DDSM_ROI, DDSM = ddsm_roi.get_roi_cropped("C:/Srp 2018/Test-ROI/CBIS-DDSM/", DDSM)
    model = applications.ResNet50(weights='imagenet', include_top=False)
    clf = joblib.load("svm_convolve_noblack.pkl")
    clf_prob = joblib.load("svm_convolve-noblack_prob.pkl")
    correct_patches = 0
    num_patches = 0
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    #DDSM = ["C:/Srp 2018/Test-ROI/CBIS-DDSM/Mass-Test_P_00017_LEFT_MLO_1/10-04-2016-DDSM-27297/1-ROI mask images-18984/000001.dcm"]
    #DDSM = ["C:/Srp 2018/Test-Full/CBIS-DDSM/Mass-Test_P_00017_LEFT_MLO/10-04-2016-DDSM-89998/1-full mammogram images-29934/000000.dcm"]
    for i in range(len(DDSM)):
        #print(i)
        pixel_array = pydicom.dcmread(DDSM[i]).pixel_array
        stem = "C:/Srp 2018/PNG_test/"+str(i)+"a"
        patch_list = convolve_svm_test(pixel_array, stem, model, clf, clf_prob)
        #print(patch_list)
        #patch list is a list of tuples that contain the x index of the middle pixel of the path
        #the y inex of the middle pixel, and the radius of the patch

        test_roi_names = get_test_roi_names(DDSM[i])
        #print(test_roi_names)
        for j in range(len(test_roi_names)):
            corrects = 0
            true_pos = 0
            false_pos = 0
            true_neg = 0
            false_neg = 0
            for k in range(len(patch_list)):
                #print("in")
                try:
                    if patch_list[k][3]==1:
                        num_patches+=1
                    coord_correct = correct_test.constructPatch(pydicom.dcmread(test_roi_names[j]).pixel_array, patch_list[k][0], patch_list[k][1], patch_list[k][2])
                    corrects, incorrects, true_pos, false_pos, true_neg, false_neg = correct_test.correct(coord_correct, patch_list[k], int(patch_list[k][3])==1)
                    if corrects!=0: break
                except AttributeError:
                    print("uh oh at "+test_roi_names[j])
            correct_patches+=corrects
            #print(corrects)
            true_positives+=true_pos
            false_positives += false_pos
            true_negatives += true_neg
            false_negatives += false_neg
    print("correct patches = "+str(correct_patches))
    print("num patches = "+str(num_patches))
    print("accuracy = "+str(correct_patches/num_patches))
    print("true positives = "+str(true_positives))
    print("false positives = "+str(false_positives))
    print("true negatives = "+str(true_negatives))
    print("false negatives = "+str(false_negatives))

def convolve_svm_test(img, stem, model, clf, clf_prob):
    row = 0
    column = 0
    iter = 0
    radius = 32
    print(img.shape)
    patch_list = []
    for multirow in range(img.shape[1]//radius):
        for multicolumn in range(img.shape[0]//radius):
            if window_percent_zeros(img[row:row+radius, column:column+radius])<=0.25:
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
                #print(out)
                new_rad = radius
                if int(out) == 1:
                    print(out)
                    init_prob = clf_prob.predict_proba(x_svm)
                    init_prob_mass = init_prob[0][1]
                    middley = int((row+radius)/2)
                    middlex = int((column+radius)/2)
                    for new_rad in range(2, radius)[::-6]:
                         true_rad = int(new_rad/2)
                         newtosave = np.zeros((224, 224))
                         newtosave[middley - true_rad:middley + true_rad, middlex - true_rad: middlex + true_rad] = img[middley - true_rad:middley + true_rad, middlex - true_rad: middlex + true_rad]
                         #above line basically has radius go from middle rather than top left
                         #print(str(row)+" , "+str(column))
                         dicomToPng(newtosave, stem+str(iter)+"b"+str(new_rad)+".png")
                         imago = image.load_img(stem+str(iter)+"b"+str(new_rad)+".png", target_size=(224, 224))
                         blah = image.img_to_array(imago)
                         blah = np.expand_dims(blah, axis=0)
                         features = model.predict(blah)
                         x_list = []
                         x_list.append(features.reshape(2048))
                         x_svm = np.asarray(x_list)
                         current_prob_mass = clf_prob.predict_proba(x_svm)[0][1]
                         if(int(current_prob_mass-init_prob_mass)==0): break
                         if current_prob_mass >= 0.9:
                            print("change in prob for image "+stem+" was "+(current_prob_mass-init_prob_mass))
                            break
                patch_list.append((int((row+radius)/2) , int((column+radius)/2), new_rad, out))
            #print(features.shape)
            column+=radius
            iter+=1
        row+=radius
        column = 0
    return patch_list



def get_test_roi_names(DDSM_name):
    list = DDSM_name.split("/")
    stem = "C:/Srp 2018/Test-ROI/CBIS-DDSM/"+list[4]
    #print(stem)
    file_list = []
    for i in range(1, 4):
        #print(stem+"_"+str(i))
        file = get_file.get_file(stem+"_"+str(i))
        if(file!=None):
            file_list.append(file)
    return file_list

def convolve_train_vgg(img, stem, model):
    row = 0
    column = 0
    itera = 0
    radius = 56
    features_nonmass = []
    #print(img.shape)
    for multirow in range(img.shape[1]//radius):
        for multicolumn in range(img.shape[0]//radius):
            if window_percent_zeros(img[row:row+radius, column:column+radius])<=0.3:
                tosave = img[row:row+radius, column:column+radius]
                print(tosave.shape)
                #print(str(row)+" , "+str(column))
                dicomToPng(tosave, stem+str(itera)+".png")
                imago = image.load_img(stem+str(itera)+".png", target_size=(56, 56))
                blah = image.img_to_array(imago)
                blah = np.expand_dims(blah, axis=0)
                features = model.predict(blah)
                #print(features.shape)
                features_nonmass.append(features)
                column+=radius
                itera+=1
        row+=radius
        column = 0
    return features_nonmass





def svm_vgg(features_nonmass, features_mass):
    non_mass_labels = np.zeros((len(features_nonmass)))
    mass_labels = np.full((len(features_mass)), 1)
    x_svm = []
    y_svm = []

    x_list = []
    x2_list = []
    for i in range(len(features_nonmass)):
        x_list.append(np.reshape(features_nonmass[i], 512))

    for j in range(len(features_mass)):
        x2_list.append(np.reshape(features_mass[j], 2048))

    x_svm = [np.concatenate((np.asarray(x_list).flatten(), np.asarray(x2_list).flatten()))]
    x_svm = np.asarray(x_svm)

    y_svm = np.concatenate((non_mass_labels.flatten(), mass_labels.flatten()))

    x_list = []
    x2_list = []
    for i in range(len(features_nonmass)):
        x_list.append(np.reshape(features_nonmass[i], 512))

    for j in range(len(features_mass)):
        x2_list.append(np.reshape(features_mass[j], 2048))

    clf = sklearn.svm.SVC()
    clf_prob = sklearn.svm.SVC(probability=True)

    clf.fit(np.asarray(x2_list), mass_labels)
    clf.fit(np.asarray(x_list), non_mass_labels)
    clf_prob.fit(np.asarray(x2_list), mass_labels)
    clf_prob.fit(np.asarray(x_list), non_mass_labels)
    """                                                                                               
    img = image.load_img("C:/Srp 2018/PNGs/nonmass494.png", target_size=(224, 224))                   
    y = np.expand_dims(image.img_to_array(img)[0:4, 0:4], axis=0)                                     
    topredict = applications.ResNet50(weights='imagenet', include_top=False).predict(y)               
    print(clf.predict(np.reshape(topredict, (1, -1))))                                                
    print("made it")                                                                                  
       """
    clf = joblib.dump(clf, "svm_vgg.pkl")
    clf_prob = joblib.dump(clf_prob, "svm_vgg_prob.pkl")
                        