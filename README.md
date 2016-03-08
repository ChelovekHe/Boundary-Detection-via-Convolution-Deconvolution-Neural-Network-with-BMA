# Boundary-Detection-via-Convolution-Deconvolution-Neural-Network-with-BMA


Time Schedule
===
DEAD LINE: Mar 13

Test baseline Mar 4

Theano & Keras

Loss function customization

Base model: Mar 9(Wed)


Code Updating
===
Data Operation Code Updated[03-05-2016]

1. Python can load images into memory and create list with element size : 481x321x3, This format can be directly used in Keras

2. Label data was in the form of *.mat. We extracted single ground truth from it into form of csv [txt](https://github.com/wuga214/Boundary-Detection-via-Convolution-Deconvolution-Neural-Network-with-BMA/blob/master/Conv-Deconv-Image-Process/data/groundTruth/train_label_flat.txt) 
  
  [Matlab Script](https://github.com/wuga214/Boundary-Detection-via-Convolution-Deconvolution-Neural-Network-with-BMA/blob/master/Conv-Deconv-Image-Process/data/groundTruth/matlabscript.m)
3. CSV label file loading in python is now functional.
4. Label reshape function works, now the data & label can be directly used by keras to do prediction!

Layer Code Updated[03-09-2016]

1. Max Pooling layer redefined under theano environment. Now the function output maxed matrix and mask!

2. Reversed Max Pooling layer is functional, it can recover the 'original' shape of matrix through max pooling mask saved

3. Convolution layer is functional, just warp of con2d in theano
