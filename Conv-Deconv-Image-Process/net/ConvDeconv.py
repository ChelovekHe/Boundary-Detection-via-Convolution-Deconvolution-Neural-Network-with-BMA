'''
Created on Mar 8, 2016

@author: Wuga
'''
import numpy
import theano
from theano import tensor as T
import layers
import files as F


class CDNN(object):

    def __init__(self, rng, batch_size, input):
        
        
        conv1 = layers.ConvLayer(
            rng,
            input=input,
            image_shape=(batch_size, 3, 480, 320),
            filter_shape=(10, 3, 3, 3),  
            padding='same'                                  
        )
        
        maxp1 = layers.MaxPooling(
            input=conv1.output,
            poolsize=(2, 2)                          
        )
         
        conv2 = layers.ConvLayer(
            rng,
            input=maxp1.output,
            image_shape=(batch_size, 10, 240, 160),
            filter_shape=(10, 10, 3, 3), 
            padding='same'                                  
        )
           
        maxp2 = layers.MaxPooling(
            input=conv2.output,
            poolsize=(2, 2)                           
        )
         
        conv3 = layers.ConvLayer(
            rng,
            input=maxp2.output,
            image_shape=(batch_size, 10, 120, 80),
            filter_shape=(10, 10, 3, 3), 
            padding='same'                                  
        )
           
        maxp3 = layers.MaxPooling(
            input=conv3.output,
            poolsize=(2, 2)                           
        )
         
        conv4 = layers.ConvLayer(
            rng,
            input=maxp3.output,
            image_shape=(batch_size, 10, 60, 40),
            filter_shape=(10, 10, 3, 3), 
            padding='same'                                  
        )
           
        maxp4 = layers.MaxPooling(
            input=conv4.output,
            poolsize=(2, 2)                           
        )
         
        conv5 = layers.ConvLayer(
            rng,
            input=maxp4.output,
            image_shape=(batch_size, 10, 30, 20),
            filter_shape=(10, 10, 3, 3), 
            padding='same'                                  
        )
           
        maxp5 = layers.MaxPooling(
            input=conv5.output,
            poolsize=(2, 2)                           
        )
         
        flat1_input = maxp5.output.flatten()
        flat1_shape = batch_size*10*15*10
          
        flat1 = layers.HiddenLayer(
            rng=rng,
            input=flat1_input,
            n_in=flat1_shape,
            n_out=flat1_shape,
            activation=T.nnet.relu
        )
          
        flat2 = layers.HiddenLayer(
            rng=rng,
            input=flat1.output,
            n_in=flat1_shape,
            n_out=flat1_shape,
            activation=T.nnet.relu
        )
          
        remax5_input = flat2.output.reshape((batch_size,10,15,10))
         
        remax5 = layers.ReverseMaxPooling(
            input=remax5_input,
            mask=maxp5.mask,
            poolsize=(2, 2)
        )
          
        deconv5 = layers.ConvLayer(
            rng,
            input=remax5.output,
            image_shape=(batch_size, 10, 30, 20),
            filter_shape=(10, 10, 3, 3),
            padding='same'                                   
        )
          
        remax4 = layers.ReverseMaxPooling(
            input=deconv5.output,
            mask=maxp4.mask,
            poolsize=(2, 2)
        )
          
        deconv4 = layers.ConvLayer(
            rng,
            input=remax4.output,
            image_shape=(batch_size, 10, 60, 40),
            filter_shape=(10, 10, 3, 3),
            padding='same'                                   
        )
          
        remax3 = layers.ReverseMaxPooling(
            input=deconv4.output,
            mask=maxp3.mask,
            poolsize=(2, 2)
        )
         
        deconv3 = layers.ConvLayer(
            rng,
            input=remax3.output,
            image_shape=(batch_size, 10, 120, 80),
            filter_shape=(10, 10, 3, 3),
            padding='same'                                   
        )
          
        remax2 = layers.ReverseMaxPooling(
            input=deconv3.output,
            mask=maxp2.mask,
            poolsize=(2, 2)
        )
         
        deconv2 = layers.ConvLayer(
            rng,
            input=remax2.output,
            image_shape=(batch_size, 10, 240, 160),
            filter_shape=(10, 10, 3, 3),
            padding='same'                                   
        )
          
        remax1 = layers.ReverseMaxPooling(
            input=deconv2.output,
            mask=maxp1.mask,
            poolsize=(2, 2)
        )
         
        deconv1 = layers.ConvLayer(
            rng,
            input=remax1.output,
            image_shape=(batch_size, 10, 480, 320),
            filter_shape=(1, 10, 3, 3),
            padding='same',
            activation=T.nnet.sigmoid                                   
        )
         
        self.y_pred=deconv1.output
        self.params=conv1.params+conv2.params+conv3.params+conv4.params+conv5.params+\
            deconv5.params+deconv4.params+deconv3.params+deconv2.params+deconv1.params   
        self.input = input
        
    def BinaryCrossEntroy(self,y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        return T.nnet.binary_crossentropy(self.y_pred, y).mean()
    
    def errors(self, y):
        idxs=(self.y_pred<0.5).nonzero()
        y_reg_pred=T.set_subtensor(self.y_pred[idxs], 0)
        idxs=(y_reg_pred>=0.5).nonzero()
        y_reg_pred=T.set_subtensor(y_reg_pred[idxs], 1)
        if y.ndim != y_reg_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', y_reg_pred.type)
            )
        return T.mean(T.neq(y_reg_pred, y))



# files='data/images/train'
# images,rotated=F.loadImage(files)
# images=numpy.array(images)
# print images.shape
# images=images.astype('float32')    
# inputs = T.ftensor4('input')
# rng = numpy.random.RandomState(1234)
# classifier = CDNN(
#         rng=rng,
#         batch_size=2,
#         input=inputs.dimshuffle((0, 3, 1, 2))
#     )
# f= theano.function([inputs],classifier.y_pred)
#         
# print f(images[:2]).shape