# Boundary-Detection-via-Convolution-Deconvolution-Neural-Network

Code Description
===
This code was generated to implement deconvolution neural network under Theano background. We found that there was no existing code to actually implement deconvolution neural network under Theano and its extension codes, even through many papers claims that they are using convolution-deconvolution structure. The only one workable code in python is under caffe environment to solve segmentation problem, in which, they actually modified caffe foundamentally. We know that most of the computer vision research is leaded in Matlab, but, for fast training using servers, python is still the optimal choice. 

Boundary detection
===
We used our network to do boundary detection task. The network does not apply optimal configuration in this experiment, and we only want to show the code is correct and workable. We highly suggest you modify the code to achieve your own task.

We also introduced a novel structure that can combine multiple network configurations, please see our paper in this positroy.
[Paper](https://github.com/wuga214/Boundary-Detection-via-Convolution-Deconvolution-Neural-Network-with-BMA/blob/master/multi-scale-boundary-3.pdf)

This is one comparision of our network(second from left) with benchmark algorithm(right most).

![png](https://github.com/wuga214/Boundary-Detection-via-Convolution-Deconvolution-Neural-Network-with-BMA/blob/master/Conv-Deconv-Image-Process/plot/compare.png)

In relatively noisy background, our implementation captures too much noise. But it is due to our network doesn't using optimal configuration. You can make it better.
Original | Ours | Benchmark | Ground Truth
------------ | ------------- | ------------- | -------------
![png](https://github.com/wuga214/Boundary-Detection-via-Convolution-Deconvolution-Neural-Network-with-BMA/blob/master/Conv-Deconv-Image-Process/plot/noisy/noise1.png)｜![png](https://github.com/wuga214/Boundary-Detection-via-Convolution-Deconvolution-Neural-Network-with-BMA/blob/master/Conv-Deconv-Image-Process/plot/noisy/noise2.png)｜![png](https://github.com/wuga214/Boundary-Detection-via-Convolution-Deconvolution-Neural-Network-with-BMA/blob/master/Conv-Deconv-Image-Process/plot/noisy/noise3.png)｜![png](https://github.com/wuga214/Boundary-Detection-via-Convolution-Deconvolution-Neural-Network-with-BMA/blob/master/Conv-Deconv-Image-Process/plot/noisy/noise4.png)

System Requirement
===
You need to install Theano and Cuda, that is all



Code Updating
===
Data Operation Code Updated[03-05-2016]

1. Python can load images into memory and create list with element size : 481x321x3, This format can be directly used in Keras

2. Label data was in the form of *.mat. We extracted single ground truth from it into form of csv [txt](https://github.com/wuga214/Boundary-Detection-via-Convolution-Deconvolution-Neural-Network-with-BMA/blob/master/Conv-Deconv-Image-Process/data/groundTruth/train_label_flat.txt) 
  
  [Matlab Script](https://github.com/wuga214/Boundary-Detection-via-Convolution-Deconvolution-Neural-Network-with-BMA/blob/master/Conv-Deconv-Image-Process/data/groundTruth/matlabscript.m)
3. CSV label file loading in python is now functional.
4. Label reshape function works, now the data & label can be directly used by keras to do prediction!

Layer Code Updated[03-07-2016]

1. Max Pooling layer redefined under theano environment. Now the function output maxed matrix and mask!

2. Reversed Max Pooling layer is functional, it can recover the 'original' shape of matrix through max pooling mask saved

3. Convolution layer is functional, just warp of con2d in theano

Network Class Updated[03-08-2016]

1. Convolution task can be worked on our code now

2. Added Keras Backend. Now the convolution layer can accept padding!!!! 

Accelerated version of Code[03-09-2016]

1. The code is able to run on GPU now! Max pooling and reverse pooling can run fast enough when training!

Complete training and evaluating[03-10-2016]

1. Forward Propagation works

2. Loss function works(binary cross entropy)

3. Data now in cPickle format(easier to load in correct shape)

4. Backpropagation works, but the error function may has some glitches... Always show 100% error

5. Code can run on Pelican server now

Bug Free Code!!!Yeah!![03-10-2016]
1. The code is fully functional and without any bug. Training and testing result seems encouraging!
![png](https://github.com/wuga214/Boundary-Detection-via-Convolution-Deconvolution-Neural-Network-with-BMA/blob/master/Conv-Deconv-Image-Process/Console.png)
