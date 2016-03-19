'''
Created on Mar 14, 2016

@author: Wuga
'''
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import sys
from numpy import genfromtxt
import backend as K
import theano
import theano.tensor as T

def loadCSV(files):
    """
    Load all images in given folder and transform them into numpy.array (3D)
    Rotate images that is in vertical format

    :type files: string type
    :param path of the folder
    """
    datapath=files
    script_dir = os.path.dirname(__file__)
    abs_data_path = os.path.join(script_dir,'..',datapath)
    print 'Loading Image Files...'
    onlyfiles = [os.path.join(abs_data_path,f) for f in listdir(abs_data_path) if isfile(join(abs_data_path, f))]
    onlyfiles = [f for f in onlyfiles if 'csv' in f]
    labels=[]
    rotated=[]
    W = theano.shared(np.ones((1,1,3,3), dtype=theano.config.floatX),borrow=True)
    for idx,f in enumerate(onlyfiles):
        lab_arr = genfromtxt(f, delimiter=',')
        m,n =lab_arr.shape
        if m<n:
            lab_arr = np.rot90(lab_arr)
            rotated.append(1)
            temp = m
            m=n
            n=temp
        else:
            rotated.append(0)
        lab_arr=lab_arr.reshape((1,1,m,n))
        input = theano.shared(lab_arr.astype(theano.config.floatX),borrow=True) 
        conv_out = K.conv2d(
            x=input,
            kernel=W,
            filter_shape=(1,1,3,3),
            image_shape=(1,1,m,n),
            border_mode='same'
        )
        lab_arr=conv_out.eval()
        lab_arr=lab_arr.reshape((m,n,1))
        labels.append(lab_arr[:-1,:-1,:]) # 640x480x3 array
        print idx*1.0/len(onlyfiles)*100,"% percent image loading complete         \r",
    print ""
    return labels,rotated

# files='data/groundTruth/train'
# images,rotated=loadCSV(files)
# print np.array(images).shape