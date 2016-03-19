'''
Created on Mar 8, 2016

@author: Wuga
'''

import numpy

import theano
import theano.tensor as T

class HiddenLayer(object):
    """
    :type rng: numpy.random.RandomState
    :param rng: a random number generator used to initialize weights

    :type input: theano.tensor.TensorType
    :param input: symbolic variable that describes the input of the
    architecture (one minibatch)

    :type n_in: int
    :param n_in: number of input units, the dimension of the space in
    which the datapoints lie

    :type n_hidden: int
    :param n_hidden: number of hidden units

    :type n_out: int
    :param n_out: number of output units, the dimension of the space in
    which the labels lie
    """
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        self.input = input
        if W is None:
            W_bound = numpy.sqrt(6. / (n_in + n_out))
            W = theano.shared(
                numpy.asarray(
                    rng.uniform(low=-W_bound, high=W_bound, size=(n_in,n_out)),
                    dtype=theano.config.floatX
                ),
                borrow=True
            )

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]
        self.input = input