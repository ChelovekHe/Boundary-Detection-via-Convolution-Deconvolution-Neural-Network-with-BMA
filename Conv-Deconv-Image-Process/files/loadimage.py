'''
Created on Mar 5, 2016

@author: Wuga
'''
import os
from os import listdir
from os.path import isfile, join
from PIL import Image 
import numpy as np
import sys

def loadImage(files):
    datapath=files
    script_dir = os.path.dirname(__file__)
    abs_data_path = os.path.join(script_dir,'..',datapath)
    print 'Loading Image Files...'
    onlyfiles = [os.path.join(abs_data_path,f) for f in listdir(abs_data_path) if isfile(join(abs_data_path, f))]
    onlyfiles = [f for f in onlyfiles if 'jpg' in f]
    images=[]
    rotated=[]
    for idx,f in enumerate(onlyfiles):
        img = Image.open(f)
        img_arr = np.array(img)
        m,n,_ =img_arr.shape
        if m<n:
            img_arr = np.rot90(img_arr)
            rotated.append(1)
        else:
            rotated.append(0)
        images.append(img_arr[:-1,:-1,:]) # 640x480x3 array
        print idx*1.0/len(onlyfiles)*100,"% percent image loading complete         \r",
    print ""
    return images,rotated

# files='data/images/train'
# images,rotated=loadImage(files)
# for x in images:
#     print x.shape
# for x in rotated:
#     print x

