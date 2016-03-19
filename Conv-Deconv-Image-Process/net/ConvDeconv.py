'''
Created on Mar 8, 2016

@author: Wuga
'''
import numpy
import theano
from theano import tensor as T
import layers
import files as F
import backend as K


class CDNN(object):

    def __init__(self, rng, batch_size, input):
        
        
        conv1_1 = layers.ConvLayer(
            rng,
            input=input,
            image_shape=(batch_size, 3, 480, 320),
            filter_shape=(32, 3, 3, 3), 
            bias=True, 
            padding='valid'                                  
        )
        
        conv1_2 = layers.ConvLayer(
            rng,
            input=K.spatial_2d_padding(conv1_1.output),
            image_shape=(batch_size, 32, 480, 320),
            filter_shape=(64, 32, 3, 3),  
            padding='valid'                                  
        )
        
        maxp1 = layers.MaxPooling(
            input=K.spatial_2d_padding(conv1_2.output),
            poolsize=(2, 2)                          
        )
           
        conv2_1 = layers.ConvLayer(
            rng,
            input=maxp1.output,
            image_shape=(batch_size, 64, 240, 160),
            filter_shape=(64, 64, 3, 3), 
            padding='valid'                                  
        )
          
        conv2_2 = layers.ConvLayer(
            rng,
            input=K.spatial_2d_padding(conv2_1.output),
            image_shape=(batch_size, 64, 240, 160),
            filter_shape=(64, 64, 3, 3), 
            padding='valid'                                  
        )
             
#         maxp2 = layers.MaxPooling(
#             input=K.spatial_2d_padding(conv2_2.output),
#             poolsize=(2, 2)                           
#         )
#           
#         conv3_1 = layers.ConvLayer(
#             rng,
#             input=maxp2.output,
#             image_shape=(batch_size, 10, 120, 80),
#             filter_shape=(10, 10, 3, 3), 
#             padding='valid'                                  
#         )
#          
#         conv3_2 = layers.ConvLayer(
#             rng,
#             input=K.spatial_2d_padding(conv3_1.output),
#             image_shape=(batch_size, 10, 120, 80),
#             filter_shape=(10, 10, 3, 3), 
#             padding='valid'                                  
#         )
#            
#         maxp3 = layers.MaxPooling(
#             input=conv3_2.output,
#             poolsize=(2, 2)                           
#         )
#          
#         conv4_1 = layers.ConvLayer(
#             rng,
#             input=maxp3.output,
#             image_shape=(batch_size, 10, 60, 40),
#             filter_shape=(10, 10, 3, 3), 
#             padding='same'                                  
#         )
#         
#         conv4_2 = layers.ConvLayer(
#             rng,
#             input=conv4_1.output,
#             image_shape=(batch_size, 10, 60, 40),
#             filter_shape=(10, 10, 3, 3), 
#             padding='same'                                  
#         )
#            
#         maxp4 = layers.MaxPooling(
#             input=conv4_2.output,
#             poolsize=(2, 2)                           
#         )
#          
#         conv5_1 = layers.ConvLayer(
#             rng,
#             input=maxp4.output,
#             image_shape=(batch_size, 10, 30, 20),
#             filter_shape=(10, 10, 3, 3), 
#             padding='same'                                  
#         )
#         
#         conv5_2 = layers.ConvLayer(
#             rng,
#             input=conv5_1.output,
#             image_shape=(batch_size, 10, 30, 20),
#             filter_shape=(10, 10, 3, 3), 
#             padding='same'                                  
#         )
#            
#         maxp5 = layers.MaxPooling(
#             input=conv5_2.output,
#             poolsize=(2, 2)                           
#         )
#           
#         flat1_input = maxp5.output.flatten(2)
#         flat1_shape = 10*15*10
#            
#         flat1 = layers.HiddenLayer(
#             rng=rng,
#             input=flat1_input,
#             n_in=flat1_shape,
#             n_out=flat1_shape,
#             activation=T.nnet.relu
#         )
#            
#         flat2 = layers.HiddenLayer(
#             rng=rng,
#             input=flat1.output,
#             n_in=flat1_shape,
#             n_out=flat1_shape,
#             activation=T.nnet.relu
#         )
#            
#         remax5_input = flat2.output.reshape((batch_size,10,15,10))
#           
#         remax5 = layers.ReverseMaxPooling(
#             input=remax5_input,
#             mask=maxp5.mask,
#             poolsize=(2, 2)
#         )
#           
#         deconv5_1 = layers.DeConvLayer(
#             rng,
#             #input=remax5.output,
#             input=conv5_2.output,
#             W=conv5_2.W,
#             image_shape=(batch_size, 10, 30, 20),
#             filter_shape=(10, 10, 3, 3),
#             padding='same'                                   
#         )
#         
#         deconv5_2 = layers.DeConvLayer(
#             rng,
#             input=deconv5_1.output,
#             W=conv5_1.W,
#             image_shape=(batch_size, 10, 30, 20),
#             filter_shape=(10, 10, 3, 3),
#             padding='same'                                   
#         )
#           
#         remax4 = layers.ReverseMaxPooling(
#             input=deconv5_2.output,
#             mask=maxp4.mask,
#             poolsize=(2, 2)
#         )
#           
#         deconv4_1 = layers.DeConvLayer(
#             rng,
#             input=remax4.output,
#             W=conv4_2.W,
#             image_shape=(batch_size, 10, 60, 40),
#             filter_shape=(10, 10, 3, 3),
#             padding='same'                                   
#         )
#         
#         deconv4_2 = layers.DeConvLayer(
#             rng,
#             input=deconv4_1.output,
#             W=conv4_1.W,
#             image_shape=(batch_size, 10, 60, 40),
#             filter_shape=(10, 10, 3, 3),
#             padding='same'                                   
#         )
#           
#         remax3 = layers.ReverseMaxPooling(
#             input=deconv4_2.output,
#             mask=maxp3.mask,
#             poolsize=(2, 2)
#         )
#          
#         deconv3_1 = layers.DeConvLayer(
#             rng,
#             #input=remax3.output,
#             input=conv3_2.output,
#             W=conv3_2.W,
#             image_shape=(batch_size, 10, 118, 78),
#             filter_shape=(10, 10, 3, 3),
#             padding='same'                                   
#         )
#          
#         deconv3_2 = layers.DeConvLayer(
#             rng,
#             input=K.spatial_2d_padding(deconv3_1.output),
#             W=conv3_1.W,
#             image_shape=(batch_size, 10, 120, 80),
#             filter_shape=(10, 10, 3, 3),
#             padding='same'                                   
#         )
#            
#         remax2 = layers.ReverseMaxPooling(
#             input=deconv3_2.output,
#             mask=maxp2.mask,
#             poolsize=(2, 2)
#         )
#           
        deconv2_1 = layers.DeConvLayer(
            rng,
            #input=remax2.output,
            input=K.spatial_2d_padding(conv2_2.output),
            W=conv2_2.W,
            image_shape=(batch_size, 64, 240, 160),
            filter_shape=(64, 64, 3, 3),
            padding='same'                                   
        )
          
        deconv2_2 = layers.DeConvLayer(
            rng,
            input=deconv2_1.output,
            W=conv2_1.W,
            image_shape=(batch_size, 64, 240, 160),
            filter_shape=(64, 64, 3, 3),
            padding='same'                                   
        )
            
        remax1 = layers.ReverseMaxPooling(
            input=deconv2_2.output,
            mask=maxp1.mask,
            poolsize=(2, 2)
        )
         
        deconv1_1 = layers.DeConvLayer(
            rng,
            input=remax1.output,
            #input=K.spatial_2d_padding(conv1_2.output),
            W= conv1_2.W,
            image_shape=(batch_size, 64, 480, 320),
            filter_shape=(32, 64, 3, 3),
            padding='same',                                  
        )
        
        deconv1_2 = layers.ConvLayer(
            rng,
            input=deconv1_1.output,
            image_shape=(batch_size, 32, 480, 320),
            filter_shape=(1, 32, 3, 3),
            bias=True,
            padding='same',
            activation=T.nnet.sigmoid                                  
        )
         
        self.y_pred=T.clip(deconv1_2.output, 0.001, 0.999)
#         self.params=conv1_1.params+conv1_2.params+conv2_1.params+conv2_2.params+\
#             conv3_1.params+conv3_2.params+conv4_1.params+conv4_2.params+conv5_1.params+conv5_2.params+\
#             deconv5_1.params+deconv5_2.params+deconv4_1.params+deconv4_2.params+deconv3_1.params+\
#             deconv3_2.params+deconv2_1.params+deconv2_2.params+deconv1_1.params+deconv1_2.params 
        self.params=conv1_1.params+conv1_2.params+conv2_1.params+conv2_2.params+deconv1_2.params
        self.input = input
        
    def BinaryCrossEntroy(self,y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        #return T.nnet.binary_crossentropy(self.y_pred, y).mean()
        #return T.nnet.categorical_crossentropy(self.y_pred, y).mean()
        return (T.pow(self.y_pred-y,2)).mean()
    
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
#         prec_idxs=(y_reg_pred>0).nonzero()
#         prec = T.mean(y[prec_idxs])
#         reca_idxs=(y>0).nonzero()
#         reca = T.mean(y_reg_pred[reca_idxs])
#         return 2*prec*reca/(prec+reca)
        


#   
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