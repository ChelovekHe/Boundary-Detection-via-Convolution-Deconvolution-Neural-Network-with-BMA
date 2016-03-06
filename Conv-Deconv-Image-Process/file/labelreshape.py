'''
Created on Mar 5, 2016

@author: Wuga
'''
import numpy as np
from file.loadimage import loadImage
from file.loadlabel import loadGroundTruth

def reshapeLables(data, rotate):
    newdata = []
    for idx,x in enumerate(data):
        if rotate[idx] == 0:
            x=x.reshape((481, 321))
        else:
            x=x.reshape((321, 481))
            x=np.rot90(x)
        newdata.append(x)
    return np.array(newdata)

image_files='data/images/train'
images,rotated=loadImage(image_files)
# for x in images:
#     print x.shape
print rotated

filename='data/groundTruth/train_label_flat.txt'
data=loadGroundTruth(filename)
print len(data[1])
print len(data[2])
print len(data)   

newlabels=reshapeLables(data, rotated)
print newlabels[1].shape
    