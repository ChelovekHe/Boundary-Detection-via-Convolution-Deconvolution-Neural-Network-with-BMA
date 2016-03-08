'''
Created on Mar 8, 2016

@author: Wuga
'''

############################
#Debug Supporting Code
############################
import numpy as np
import theano
from theano import tensor as T
from theano.tensor.nnet import conv

class ReverseMaxPooling(object):
    
    def __init__(self, mask, poolsize=(2, 2), ignore_border=True):
        self.poolsize = poolsize
        self.ignore_border = ignore_border
        self.mask_index=mask.nonzero()
        self.mask_flatten=mask[self.mask_index]

M=np.array([[[[1, 0, 0, 0, 0, 1, 0, 1, 0, 0],
[0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
[0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
[1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
[0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 1, 0, 0, 1, 1, 0],
[0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
[1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
[1, 0, 0, 0, 0, 1, 1, 0, 0, 0],
[0, 0, 1, 0, 0, 0, 0, 0, 1, 0]]]])
print M.shape   
mask = T.dtensor4('mask')
X=np.random.rand(1,1,5,5)
print X
layer=ReverseMaxPooling( mask)
f= theano.function([mask],layer.mask_flatten)
print f(X)