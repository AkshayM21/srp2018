# Synopsys 2018 Project
A Novel Application of Convolutional Neural Networks to Segment Breast Cancer Tumors in Mammograms


Made by Akshay Manglik and Andy Lee

Apologies in advance to anyone reading through this code for the lack of documentation.

This project was made using Pydicom, Keras, numpy, scipy, scikit-learn, OpenCV, scikit-image, pickle, and a few other libraries. We used the CBIS-DDSM dataset curated by Prof. Daniel Rubin and colleagues at Stanford, available from TCIA here: https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM

## File descriptions

### resnetsvm.py
This file contains all functions dealing with machine learning models that were used in this project. The following are descriptions of the functions in this file:
#### getFeatures
This function put the mass and nonmass image data through the ResNet50 architecture (lacking the fully connected layer) so it outputs and saves feature vectors for both mass and nonmass. Those are saved using pickle.
#### get_features_vgg
Like above, but using the VGG-19 model instead of ResNet50. Overall we determined using Resnet was better.
#### get_features_convolve
Here, instead of putting in the 224x224 image, we broke it up into smaller 56x56 subimages. Doing this allowed for better prediction by the SVM in the end.
#### dicomToPng
simple utility function that converted dicom files to png images for easy checking during testing
#### logistic_reg
We also tried using a logistic regression model instead of an SVM -- overall we determined using the SVM was better
#### svm
Uses a Support Vector Machine as classifer. We also had a separate SVM model with the probability attribute set as true which allowed us to see the SVM's own estimation of the accuracy of its classification.
#### predict_model
Used during testing to get the SVM's predictions of an input image as containing mass or not
#### convolve_train_noblack
Breaks up the image into smaller images, which are processed by the model (either ResNet or VGG) into feature vectors. Here, noblack means it excludes images that are fully or mostly black pixels, as that can cause overfitting problems for the SVM due to the number of black images.
#### window_percent_zeros
tells us what percentage of the image is black / has a pixel value of zero
#### convolve_train
Like convolve_train_noblack excecpt it does not exclude fully or mostly black images, which led to problems.
#### test
Used to determine the TP, TN, FP, and FN stats of our pipeline by applying to to unseen mammograms.
#### convolve_svm_test
This was used during testing. Essentially, using the probability measure of the svm, we would expand or contract the size of the patch (and thus include or exclude certain pixels) until it reached the highest probability of accuracy in its output. These images were then compared to the expert segmentations provided in the training set to see if the model was correct or not.
#### get_test_roi_names
Utility function that returns a list of the filepaths of the expert segmentations the mammograms in the test set.
#### convolve_train_vgg
Like convolve_train except using the VGG model instead of ResNet
#### svm_vgg
Like the SVM method except used in conjunction with VGG-19 not ResNet50 for generating feature vectors from the images.

### DDSM_sort.py
The functions in this file were used to narrow our dataset programmatically so that it only included mammograms taken from the  mediolateral oblique (MLO) view, and not from the craniocaudal (CC) view.

### artifact_removal.py
The functions here were our implementation of Otsu's method to isolate background artifacts from the foreground breast. Due to time and speed concerns, we eventually opted with OpenCv's Otsu's method function instead.

### correct_test.py
This waS used to determine if any patches that the model had determined to include mass did indeed contain mass. The construct patch method was used to generate an array that contained the patch data given x, y, and radius data. The correct function determined if the patch was a TP, TN, FP, or FN.

### ddsm_roi.py
This was a set of utility functions that listed the file paths of the segmented regions of interest from the data set that correspond with each whole mammogram.

### extra_pectoral_muscle.py

### flip.py
This would flip the image sideways. It was used to standardize all mammograms as being in the LEFT view.

### get_file.py
A list of utility functions for opbtaining the partial or full paths of a mammogram file due to the unique file structure of the dataset.

### main.py
Contains the post-preprocessing function calls in order -- ie, the ones dealing with the ML models. Most are commented out as during testing we only needed to run a few at a time.

### mass_v_nonmass.py
Since the segmentations in the dataset were black and white masks, this file was used to apply those masks to the original image to obtain an image of the mass alone in the mammogram, also part of preprocessing.

### median_noise.py
A preprocessing algorithm used to reduce noise. It would iterate a 3x3 square around the image, and the center pixel would be replaced with the median value of the pixels in the square. This was an effective noise reduction method according to papers reviewed prior to the project.

### pectoral_muscle.py and extra_pectoral_muscle.py
Used for isolating the breast from the pectoral muscle in the mammogram, as that can confound results. extra_pectoral_muscle.py contains the previous pectoral muscle functions that we had made but were rendered useless when we moved to canny edge detection and straight line approximation instead of area approximation with connected components labeling.

### preprocessing.py
Contains the preprocessing method calls in order. Ultimatley returns an array of mass image data (array of pixel arrays) and an array of nonmass image data. These were then used for training the model.

### replace.py
Simple utility function that switches the pixel data of two dicom files.

### .p and .pkl files
These were the pickle files. .p files are pickled arrays of image data or feature vectors. .pkl are pickled models with varying characteristics/parameters.

## Results
Overall, 96% true positive rate but only 57% overall accuracy due to a large number of false negatives -- still shows promise for the model.
