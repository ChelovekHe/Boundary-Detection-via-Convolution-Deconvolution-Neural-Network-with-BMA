'''
Created on Mar 8, 2016

@author: Wuga
'''

import numpy

import theano
import theano.tensor as T

class HiddenLayer(object):
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