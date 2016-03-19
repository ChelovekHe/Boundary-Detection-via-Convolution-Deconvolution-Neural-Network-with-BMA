'''
Created on Mar 5, 2016

@author: Wuga
'''
import numpy as np
from files.loadimage import loadImage
from files.loadlabel import loadGroundTruth

def reshapeLables(data, rotate):
    """
    Reshape target labels into 480*320 shape

    :type data: numpy.array (2D)
    :param data: ground true labels in flat shape

    :type rotate: boolean
    :param rotate: reshape according to the original image shape
    """
    newdata = []
    for idx,x in enumerate(data):
        if rotate[idx] == 0:
            x=x.reshape((481, 321, 1))
        else:
            x=x.reshape((321, 481, 1))
            x=np.rot90(x)
        newdata.append(x[:-1,:-1,:])
        print idx*1.0/len(data)*100,"% percent reshape complete         \r",
    print ""
    return np.array(newdata)

    ######################
    # Function Test code #
    ######################

# image_files='data/images/train'
# images,rotated=loadImage(image_files)
# # for x in images:
# #     print x.shape
# print rotated
#  
# filename='data/groundTruth/train_label_flat.txt'
# data=loadGroundTruth(filename)
# print len(data[1])
# print len(data[2])
# print len(data)   
#  
# newlabels=reshapeLables(data, rotated)
# print newlabels[1].shape
    