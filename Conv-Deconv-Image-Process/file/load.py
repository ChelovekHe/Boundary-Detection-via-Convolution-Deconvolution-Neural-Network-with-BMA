'''
Created on Mar 5, 2016

@author: Wuga
'''

import os
from os import listdir
from os.path import isfile, join
from PIL import Image 
import numpy as np

def load(files):
    datapath=files
    script_dir = os.path.dirname(__file__)
    abs_data_path = os.path.join(script_dir,'..',datapath)
    print abs_data_path
    onlyfiles = [os.path.join(abs_data_path,f) for f in listdir(abs_data_path) if isfile(join(abs_data_path, f))]
    onlyfiles = [f for f in onlyfiles if 'jpg' in f]
    images=[]
    for f in onlyfiles:
        img = Image.open(f)
        img_arr = np.array(img)
        m,n,_ =img_arr.shape
        if m<n:
            img_arr = np.rot90(img_arr)
        images.append(img_arr) # 640x480x3 array
    return images

files='data/images/train'
images=load(files)
for x in images:
    print x.shape

