#!/usr//bin/env python
# -*- coding: utf-8; mode: python; -*-

"""Module providing class for meta-classification.

Attributes:
BaseJudge (class):
  class for joining decisions of single classifiers

"""

##################################################################
# Imports
from __future__ import absolute_import, print_function

from dsenser.theano_utils import floatX, rmsprop, HE_UNIFORM, MAX_ITERS

from datetime import datetime
from lasagne.init import HeUniform, Orthogonal
from theano import config, tensor as TT
import numpy as np
import sys
import theano

##################################################################
# Variables and Constants
EPS = 0.  # 1e-3
CONV_EPS = 1e-5
INF = float("inf")


##################################################################
# Class
class BaseJudge(object):
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
        self.y_bias = theano.shared(value=HE_UNIFORM((1, self.n_y)),
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
        # gold vector
        y_gold = TT.dvector(name="y_gold")
        # define cost and optimization function
        cost = TT.sum((self.y_pred - y_gold) ** 2)
        # predict = theano.function([self.x, y_gold], [self.y_pred, cost],
        #                           name="predict")
        gradients = TT.grad(cost, wrt=self._params)
        f_grad_shared, f_update, _ = rmsprop(self._params, gradients,
                                             [self.x], y_gold, cost)
        # perform actual training
        min_cost = INF
        best_params = []
        start_time = end_time = None
        time_delta = prev_icost = icost = 0.
        a_ts = [(floatX(x), floatX(y)) for x, y in a_ts]
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
            if icost < min_cost:
                best_params = [p.get_value() for p in self._params]
                min_cost = icost
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
        # set best values seen during training
        if best_params:
            for p, val in zip(self._params, best_params):
                p.set_value(val)

    def predict(self, a_x):
        """Method for predicting sense of single relation.

        Args:
        a_x (np.array):
          (submodels x class) array of input predictions

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
        return theano.shared(value=floatX(tens),
                             name="X2Y")
