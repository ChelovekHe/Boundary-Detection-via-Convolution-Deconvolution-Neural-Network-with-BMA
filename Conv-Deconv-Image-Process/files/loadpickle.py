'''
Created on Mar 10, 2016

@author: Wuga
'''

import os
import numpy
import theano
import cPickle as pickle
import gzip
from PIL import Image 

def loadPickleData(dataset):

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path):
            dataset = new_path

    print('... loading data')

    # Load the dataset
    with open(dataset, 'rb') as pickle_file:
        train_set, valid_set, test_set = pickle.load(pickle_file)
        
    print numpy.array(valid_set[0]).shape
    print numpy.array(valid_set[1]).shape

    def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        norm_data_x = (numpy.asarray(data_x,dtype=theano.config.floatX)-127.5)/255.0
        norm_data_y = numpy.asarray(data_y,dtype=theano.config.floatX)
        shared_x = theano.shared(norm_data_x,
                                 borrow=borrow)
        shared_y = theano.shared(norm_data_y,
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, shared_y

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

    ######################
    # Function Test code #
    ######################
    
# datasets = loadPickleData('data.save')
# test_set_x, test_set_y = datasets[2]
# predicted_values = test_set_y.get_value()
# predicted_values = numpy.array(predicted_values)
# predicted_values=predicted_values[:10]
# print("Predicted values for the first 10 examples in test set:")
# for idx,I in enumerate(predicted_values):
#     I8 = (I.reshape((480,320)) * 255).astype(numpy.uint8)
#     rgbArray = numpy.zeros((480,320,3), 'uint8')
#     rgbArray[..., 0] = I8
#     rgbArray[..., 1] = I8
#     rgbArray[..., 2] = I8
#     img = Image.fromarray(rgbArray)
#     img.save('/Users/Wuga/Documents/workspace/'+str(idx)+'myimg.jpeg')