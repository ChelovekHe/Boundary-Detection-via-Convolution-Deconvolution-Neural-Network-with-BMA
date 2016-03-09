'''
Created on Mar 5, 2016

@author: Wuga
'''
import os
import csv
import numpy as np

def loadGroundTruth(filename):
    datapath=filename
    script_dir = os.path.dirname(__file__)
    abs_data_path = os.path.join(script_dir, '..',datapath)
    data=np.genfromtxt(abs_data_path,delimiter=',')
    return data

# filename='data/groundTruth/train_label_flat.txt'
# data=loadGroundTruth(filename)
# print len(data[1])
# print len(data[2])
# print len(data)