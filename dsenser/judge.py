#!/usr//bin/env python
# -*- coding: utf-8; mode: python; -*-

"""Module providing class for meta-classification.

Attributes:
Judge (class):
  class for joining decisions of single classifiers

"""

##################################################################
# Imports
from __future__ import absolute_import, print_function

from datetime import datetime
from lasagne.init import HeUniform, Orthogonal
from theano import config, tensor as TT
import numpy as np
import sys
import theano

##################################################################
# Variables and Constants
HE_UNIFORM = HeUniform()
EPS = 1e-3
CONV_EPS = 1e-5
MAX_ITERS = 60


##################################################################
# Variables and Constants
def _floatX(a_data, a_dtype=config.floatX):
    """Return numpy array populated with the given data.

    Args:
    data (np.array):
      input tensor
    dtype (class):
      digit type

    Returns:
    (np.array):
     array populated with the given data

    """
    return np.asarray(a_data, dtype=a_dtype)


def rmsprop(tparams, grads, x, y, cost):
    """
    A variant of  SGD that scales the step size by running average of the
    recent step norms.

    Parameters:
    tpramas (Theano SharedVariable):
        Model parameters
    grads (Theano variable):
        Gradients of cost w.r.t to parameres
    x (Theano variable):
        Model inputs
    y (Theano variable):
        Targets
    cost (Theano variable):
        Objective fucntion to minimize

    Notes:
    For more information, see [Hint2014]_.

    .. [Hint2014] Geoff Hinton, *Neural Networks for Machine Learning*,
       lecture 6a,
       http://cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf

    """
    zipped_grads = [theano.shared(p.get_value() * _floatX(0.))
                    for p in tparams]
    running_grads = [theano.shared(p.get_value() * _floatX(0.))
                     for p in tparams]
    running_grads2 = [theano.shared(p.get_value() * _floatX(0.))
                      for p in tparams]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, y], cost,
                                    updates=zgup + rgup + rg2up,
                                    name='rmsprop_f_grad_shared')

    updir = [theano.shared(p.get_value() * _floatX(0.))
             for p in tparams]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / TT.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(tparams, updir_new)]
    f_update = theano.function([], [], updates=updir_new + param_up,
                               on_unused_input='ignore',
                               name='rmsprop_f_update')
    return (f_grad_shared, f_update, (zipped_grads, running_grads,
                                      running_grads2, updir))


##################################################################
# Class
class Judge(object):
    """Meta-classifier.

    This classifier unites decisions of other multiple independent classifiers.

    Attrs:

    Methods:

    """

    def __init__(self, a_n_x, a_n_y):
        """Class constructor.

        Args:
        a_n_x (int):
          number of underlying cassifiers
        a_n_y (int):
          number of classes to predict


        """
        self.n_x = a_n_x
        self.n_y = a_n_y
        # define the network
        # input matrix
        self.x = TT.dmatrix(name="x")
        # mapping from input to output vector
        self.X2Y = self._init_X2Y()
        self.y_bias = theano.shared(value=HE_UNIFORM.sample((1, self.n_y)),
                                    name="y_bias")
        # prediction vector
        self.y_pred = TT.nnet.softmax(
            TT.tensordot(self.x, self.X2Y, ((1, 0), (2, 1))) + self.y_bias)
        # predicted label
        self.y_lbl = TT.argmax(self.y_pred, axis=1)[0]
        self._predict = theano.function([self.x],
                                        [self.y_lbl, self.y_pred],
                                        name="predict")
        # define trainable parameters
        self._params = [self.X2Y, self.y_bias]

    def train(self, a_ts, a_dev_data=None):
        """Method for training the model.

        Args:
        a_ts (list(2-tuple(x, y))):
          list of training JSON data
        a_dev_data (2-tuple(dict, dict) or None):
          list of development JSON data

        Returns:
        (void)

        """
        print("Training judge model ...", file=sys.stderr)
        # gold vector
        y_gold = TT.dvector(name="y_gold")
        # define cost and optimization function
        cost = TT.sum((self.y_pred - y_gold) ** 2)
        # predict = theano.function([self.x, y_gold], [self.y_pred, cost],
        #                           name="predict")
        gradients = TT.grad(cost, wrt=self._params)
        f_grad_shared, f_update, _ = rmsprop(self._params, gradients,
                                             self.x, y_gold, cost)
        # perform actual training
        start_time = end_time = None
        time_delta = prev_icost = icost = 0.
        a_ts = [(_floatX(x), _floatX(y)) for x, y in a_ts]
        for i in xrange(MAX_ITERS):
            icost = 0.
            np.random.shuffle(a_ts)
            start_time = datetime.utcnow()
            for x_i, y_i in a_ts:
                try:
                    icost += f_grad_shared(x_i, y_i)
                    f_update()
                except Exception as e:
                    raise e
            end_time = datetime.utcnow()
            time_delta = (end_time - start_time).seconds
            print(
                "Iteration #{:d}: cost = {:f} ({:.2f} sec)".format(i,
                                                                   icost,
                                                                   time_delta),
                file=sys.stderr)
            if abs(prev_icost - icost) < CONV_EPS:
                break
            prev_icost = icost
        print("done", file=sys.stderr)

    def predict(self, a_x):
        """Method for predicting sense of single relation.

        Args:
        a_x (np.array):
          (submodels x class) array of submodels' predictions

        Returns:
        str:
          most probable sense of discourse relation

        """
        return self._predict(a_x)

    def _init_X2Y(self):
        """Initialize tensor for mapping input mtx to output vec.

        Args:
        (void)

        Returns:
        (theano.shared):
          shared theano tensor

        """
        tens = np.zeros((self.n_y, self.n_x, self.n_y))
        tens -= EPS
        for i in xrange(self.n_y):
            tens[i, :, i] = 1.
        return theano.shared(value=_floatX(tens),
                             name="X2Y")
