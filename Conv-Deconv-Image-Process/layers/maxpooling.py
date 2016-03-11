'''
Created on Mar 7, 2016

@author: Wuga
'''
from theano.tensor.signal import pool
import theano.tensor as T
import numpy as np
import theano

class MaxPooling(object):
    """Pool Layer of a convolutional network """

    def __init__(self, input, poolsize=(2, 2), ignore_border=False):
        
        self.input = input
        self.poolsize = poolsize
        self.ignore_border = ignore_border

        pooled_out = pool.pool_2d(
            input=self.input,
            ds=self.poolsize,
            ignore_border=self.ignore_border,
            mode='max'
        )
        self.output = pooled_out
        self.mask = T.grad(None, wrt=self.input, known_grads={pooled_out: T.ones_like(pooled_out)})

#         s1 = self.poolsize[0]
#         s2 = self.poolsize[1]
#         self.output = pooled_out
#         recovered= self.output.repeat(s1, axis=2).repeat(s2, axis=3)
#         mask = T.zeros(recovered.shape)
#         shp = input.shape
#         mask = T.set_subtensor(mask[:,:,:shp[2],:shp[3]], input)
#         mask = T.eq(mask,recovered)
# 
#         self.mask = mask[:,:,:-1,:-1]
#         self.input = input

# input = T.dtensor4('input')
# layer=MaxPooling(input)
# f= theano.function([input],layer.mask)
# X=np.random.rand(2,3,10,10)
# print X
# print f(X)