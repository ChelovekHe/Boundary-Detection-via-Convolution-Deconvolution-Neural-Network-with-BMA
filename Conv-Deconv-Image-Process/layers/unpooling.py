'''
Created on Mar 7, 2016

@author: Wuga
'''

import numpy as np
import theano
from theano import tensor as T
from theano.tensor.nnet import conv

class ReverseMaxPooling(object):
    
    def __init__(self, input, mask, poolsize=(2, 2), ignore_border=True):
        self.input = input
        self.poolsize = poolsize
        self.ignore_border = ignore_border
        s1 = self.poolsize[0]
        s2 = self.poolsize[1]
        recovered= self.input.repeat(s1, axis=2).repeat(s2, axis=3)
        output= recovered*mask
        self.output=output

# M=np.array([[[[1, 0, 0, 0, 0, 1, 0, 1, 0, 0],
# [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
# [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
# [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
# [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
# [0, 0, 0, 0, 1, 0, 0, 1, 1, 0],
# [0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
# [1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
# [1, 0, 0, 0, 0, 1, 1, 0, 0, 0],
# [0, 0, 1, 0, 0, 0, 0, 0, 1, 0]]]])
# print M.shape   
# X=np.random.rand(1,1,5,5)
# print X
# input = T.dtensor4('input')
# mask = T.dtensor4('mask')
# layer=ReverseMaxPooling(input, mask)
# f= theano.function([input,mask],layer.output)
# print f(X,M)