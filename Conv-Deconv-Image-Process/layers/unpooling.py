'''
Created on Mar 7, 2016

@author: Wuga
'''

import numpy as np
import theano
from theano.tensor.signal import pool
from theano import tensor as T
from theano.tensor.nnet import conv

class ReverseMaxPooling(object):
    '''
    :type rng: numpy.random.RandomState
    :param rng: a random number generator used to initialize weights
    
    :type input: theano.tensor.dtensor4
    :param input: symbolic variable that describes the input of the
    architecture (one minibatch)
    
    :type mask: theano.tensor.dtensor4
    :param mask: symbolic variable that describes the input of the
    architecture (one minibatch)
    
    :type poolsize: tuple(int,int)
    :param poolsize: pooling size of max pooling
    
    :type ignore_border: boolean
    :param ignore_border: ignore border pixels when doing max pooling
    '''
    def __init__(self, input, mask, poolsize=(2, 2), ignore_border=True):
        self.input = input
        self.poolsize = poolsize
        self.ignore_border = ignore_border
        
        mask_pooled_out = pool.pool_2d(
            input=mask,
            ds=self.poolsize,
            ignore_border=self.ignore_border,
            mode='max'
        )
        
        #Use symbolic programming property to reshape layers with mask 
        #and maxpooled  layer
        self.output = T.grad(None, wrt=mask, known_grads={mask_pooled_out: input})
        self.input = input
        
    ######################
    # Function Test code #
    ######################

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