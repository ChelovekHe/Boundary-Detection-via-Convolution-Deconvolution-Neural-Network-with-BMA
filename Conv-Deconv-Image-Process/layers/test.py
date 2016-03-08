'''
Created on Mar 7, 2016

@author: Wuga
'''

############################
#Debug Supporting Code
############################

import numpy as np
import theano
import theano.tensor as T

minibatchsize = 2
numfilters = 3
numsamples = 4
upsampfactor = 5

totalitems = minibatchsize * numfilters * numsamples

code = np.arange(totalitems).reshape((minibatchsize, numfilters, numsamples))

auxpos = np.arange(totalitems).reshape((minibatchsize, numfilters, numsamples)) % upsampfactor 
auxpos += (np.arange(4) * 5).reshape((1,1,-1))

# first in numpy
shp = code.shape
upsampled_np = np.zeros((shp[0], shp[1], shp[2] * upsampfactor))
upsampled_np[np.arange(shp[0]).reshape(-1, 1, 1), np.arange(shp[1]).reshape(1, -1, 1), auxpos] = code

print "numpy output:"
print upsampled_np

# now the same idea in theano
encoded = T.tensor3()
positions = T.tensor3(dtype='int64')
shp = encoded.shape
upsampled = T.zeros((shp[0], shp[1], shp[2] * upsampfactor))
upsampled = T.set_subtensor(upsampled[T.arange(shp[0]).reshape((-1, 1, 1)), T.arange(shp[1]).reshape((1, -1, 1)), positions], encoded)

print auxpos
print "theano output:"
print upsampled.eval({encoded: code, positions: auxpos})