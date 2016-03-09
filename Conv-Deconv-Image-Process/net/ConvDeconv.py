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
        
        self.layer1 = layers.ConvLayer(
            rng,
            input=input,
            image_shape=(batch_size, 3, 481, 321),
            filter_shape=(10, 3, 3, 3),  
            padding='same'                                  
        )
        
        self.layer2 = layers.MaxPooling(
            input=self.layer1.output,
            poolsize=(2, 2)                          
        )
        
        self.layer3 = layers.ConvLayer(
            rng,
            input=self.layer2.output,
            image_shape=(batch_size, 10, 241, 161),
            filter_shape=(10, 10, 3, 3), 
            padding='same'                                  
        )
          
        self.layer4 = layers.MaxPooling(
            input=self.layer3.output,
            poolsize=(2, 2)                           
        )
        
        self.layer5 = layers.ConvLayer(
            rng,
            input=self.layer4.output,
            image_shape=(batch_size, 10, 121, 81),
            filter_shape=(10, 10, 3, 3), 
            padding='same'                                  
        )
          
        self.layer6 = layers.MaxPooling(
            input=self.layer5.output,
            poolsize=(2, 2)                           
        )
        
        self.layer7 = layers.ConvLayer(
            rng,
            input=self.layer6.output,
            image_shape=(batch_size, 10, 61, 41),
            filter_shape=(10, 10, 3, 3), 
            padding='same'                                  
        )
          
        self.layer8 = layers.MaxPooling(
            input=self.layer7.output,
            poolsize=(2, 2)                           
        )
        
        layer9_input = self.layer8.output.flatten()
        layer9_input_shape = batch_size*10*31*21
         
        self.layer9 = layers.HiddenLayer(
            rng=rng,
            input=layer9_input,
            n_in=layer9_input_shape,
            n_out=layer9_input_shape,
            activation=T.tanh
        )
         
        self.layer10 = layers.HiddenLayer(
            rng=rng,
            input=self.layer9.output,
            n_in=layer9_input_shape,
            n_out=layer9_input_shape,
            activation=T.tanh
        )
         
        self.layer11_input = self.layer10.output.reshape((batch_size,10,31,21))
         
        self.layer11 = layers.ReverseMaxPooling(
            input=self.layer11_input,
            mask=self.layer8.mask,
            poolsize=(2, 2)
        )
         
        self.layer12 = layers.ConvLayer(
            rng,
            input=self.layer11.output,
            image_shape=(batch_size, 10, 61, 41),
            filter_shape=(10, 10, 3, 3),
            padding='same'                                   
        )
         
        self.layer13 = layers.ReverseMaxPooling(
            input=self.layer12.output,
            mask=self.layer6.mask,
            poolsize=(2, 2)
        )
        
        self.layer14 = layers.ConvLayer(
            rng,
            input=self.layer13.output,
            image_shape=(batch_size, 10, 121, 81),
            filter_shape=(5, 10, 3, 3),
            padding='same'                                   
        )
         
        self.layer15 = layers.ReverseMaxPooling(
            input=self.layer14.output,
            mask=self.layer4.mask,
            poolsize=(2, 2)
        )
        
        self.layer16 = layers.ConvLayer(
            rng,
            input=self.layer15.output,
            image_shape=(batch_size, 5, 241, 161),
            filter_shape=(1, 5, 3, 3),
            padding='same'                                   
        )
         
        self.layer17 = layers.ReverseMaxPooling(
            input=self.layer16.output,
            mask=self.layer4.mask,
            poolsize=(2, 2)
        )
        
        self.layer18 = layers.ConvLayer(
            rng,
            input=self.layer17.output,
            image_shape=(batch_size, 1, 481, 321),
            filter_shape=(1, 5, 3, 3),
            padding='same'                                   
        )
        
        self.prediction = self.layer17.output
        self.input = input


files='data/images/train'
images,rotated=F.loadImage(files)
images=numpy.transpose(images, (0,3,1,2))

images=images.astype('float32')    
inputs = T.ftensor4('input')
rng = numpy.random.RandomState(1234)
classifier = CDNN(
        rng=rng,
        batch_size=1,
        input=inputs
    )
f= theano.function([inputs],classifier.prediction)
    
print f(images[:1]).shape