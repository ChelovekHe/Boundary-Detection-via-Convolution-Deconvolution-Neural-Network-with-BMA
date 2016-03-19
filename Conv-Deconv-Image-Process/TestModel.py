'''
Created on Mar 5, 2016

This code is one extension of theano sample code

@author: Wuga
'''
import numpy
import theano
import theano.tensor as T
import layers
import files as F
from net import CDNN
import timeit
import cPickle
from PIL import Image 

def ModelTester(learning_rate=0.01, n_epochs=100, batch_size=40):
    """
    The function test the proposed model on BSDS500 dataset in pickle format
    (Data transform code is available in files folder of this package)
    
    This function will also train and save best model!

    :type learning_rate: float type
    :param learning_rate: gradient descent learning rate
    
    :type n_epochs: integer type
    :param n_epochs: number of epochs to run
    
    :type batch_size: integer type
    :param batch_size: size of data in each loop of gradient decent
    """
    
    datasets = F.loadPickleData('data.save')
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
 
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size
    
    train_set_x=train_set_x.dimshuffle((0, 3, 1, 2))
    train_set_y=train_set_y.dimshuffle((0, 3, 1, 2))
    valid_set_x=valid_set_x.dimshuffle((0, 3, 1, 2))
    valid_set_y=valid_set_y.dimshuffle((0, 3, 1, 2))
    test_set_x=test_set_x.dimshuffle((0, 3, 1, 2))
    test_set_y=test_set_y.dimshuffle((0, 3, 1, 2))

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    index = T.lscalar()  
    x = T.ftensor4('x')  
    y = T.ftensor4('y')
    

    rng = numpy.random.RandomState(1234)

    classifier = CDNN(
        rng=rng,
        batch_size=batch_size,
        input=x
    )
    
    cost = classifier.BinaryCrossEntroy(y)
    
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )
    
    test_cost_model = theano.function(
        inputs=[index],
        outputs=classifier.BinaryCrossEntroy(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )
    
    validate_cost_model = theano.function(
        inputs=[index],
        outputs=classifier.BinaryCrossEntroy(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )
    
    train_cost_model = theano.function(
        inputs=[index],
        outputs=classifier.BinaryCrossEntroy(y),
        givens={
            x: train_set_x[index * batch_size:(index + 1) * batch_size],
            y: train_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )
    
    train_error_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: train_set_x[index * batch_size:(index + 1) * batch_size],
            y: train_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )
    
    
    gparams = [T.grad(cost, param) for param in classifier.params]

    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    
    ###############
    # TRAIN MODEL #
    ###############
    print('... training')

    patience = 10000   
    validation_frequency = min(n_train_batches, patience // 2)
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False
    
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                train_losses = [train_error_model(i) for i
                                     in range(n_train_batches)]
                this_train_losses = numpy.mean(train_losses)
                
                train_cost = [train_cost_model(i) for i
                                     in range(n_train_batches)]
                this_train_cost = numpy.mean(train_cost)

                print(
                    'epoch %i, minibatch %i/%i, train error %f %%, train cost %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_train_losses *100,
                        this_train_cost * 100.
                    )
                )

            if patience <= iter:
                done_looping = True
                break
    end_time = timeit.default_timer()
    
    # compile a predictor function
    predict_model = theano.function(
        inputs=[classifier.input],
        outputs=classifier.y_pred)
    test_set_x = test_set_x.eval()
    predicted_values = predict_model(test_set_x[:5])
    print("Predicted values for the first 10 examples in test set:")
    print (predicted_values[0].reshape((480,320))*255).astype(numpy.uint8)
    for idx,I in enumerate(predicted_values):
        I8 = (I.reshape((480,320)) * 255).astype(numpy.uint8)
        rgbArray = numpy.zeros((480,320,3), 'uint8')
        rgbArray[..., 0] = I8
        rgbArray[..., 1] = I8
        rgbArray[..., 2] = I8
        img = Image.fromarray(rgbArray)
        img.save(str(idx)+'myimg.jpeg')
    predicted_values = predict_model(test_set_x[5:10])
    print("Predicted values for the first 10 examples in test set:")
    print (predicted_values[0].reshape((480,320))*255).astype(numpy.uint8)
    for idx,I in enumerate(predicted_values):
        I8 = (I.reshape((480,320)) * 255).astype(numpy.uint8)
        rgbArray = numpy.zeros((480,320,3), 'uint8')
        rgbArray[..., 0] = I8
        rgbArray[..., 1] = I8
        rgbArray[..., 2] = I8
        img = Image.fromarray(rgbArray)
        img.save(str(idx)+'myimg.jpeg')

    
ModelTester(learning_rate=0.1, n_epochs=200, batch_size=5)
