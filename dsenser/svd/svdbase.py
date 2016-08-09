#!/usr/bin/env python
# -*- coding: utf-8; mode: python; -*-

"""Module providing abstract interface class for SVD sense classification.

Attributes:
  SVDBaseSenser (class):
    abstract class defining interface for explicit and implicit classifier

"""

##################################################################
# Imports
from __future__ import absolute_import, print_function

from dsenser.nnbase import NNBaseSenser
from dsenser.theano_utils import floatX, theano, HE_UNIFORM, ORTHOGONAL, TT

import numpy as np

##################################################################
# Variables and Constants
MIN_DIM = 50


##################################################################
# Class
class SVDBaseSenser(NNBaseSenser):
    """Abstract class for disambiguating relation senses.

    Attributes:
      n_y (int): number of distinct classes

    """

    def _init_nn(self):
        """Initialize neural network.

        """
        self.intm_dim = max(MIN_DIM, self.ndim - (self.ndim - self.n_y) / 2)
        # indices of word embeddings
        self.W_INDICES_ARG1 = TT.ivector(name="W_INDICES_ARG1")
        self.W_INDICES_ARG2 = TT.ivector(name="W_INDICES_ARG2")
        # connective's index
        self.CONN_INDEX = TT.iscalar(name="CONN_INDEX")
        # initialize the matrix of word embeddings
        self.init_w_emb()
        # word embeddings of the arguments
        self.EMB_ARG1 = self.W_EMB[self.W_INDICES_ARG1]
        self.EMB_ARG2 = self.W_EMB[self.W_INDICES_ARG2]
        # connective's embedding
        self._init_conn_emb()
        self.EMB_CONN = self.CONN_EMB[self.CONN_INDEX]
        # perform matrix decomposition
        _, _, self.ARG1 = TT.nlinalg.svd(self.EMB_ARG1,
                                         full_matrices=True)
        _, _, self.ARG2 = TT.nlinalg.svd(self.EMB_ARG2,
                                         full_matrices=True)
        self.ARG_DIFF = self.ARG1 - self.ARG2
        # map decomposed matrices to the intermediate level
        self.ARG_DIFF2I = theano.shared(value=HE_UNIFORM((self.ndim, 1)),
                                        name="ARG_DIFF2I")
        self.arg_diff_bias = theano.shared(value=HE_UNIFORM((1, self.ndim)),
                                           name="arg_diff_bias")
        self._params.extend([self.ARG_DIFF2I, self.arg_diff_bias])
        self.ARGS = (TT.dot(self.ARG_DIFF, self.ARG_DIFF2I).T +
                     self.arg_diff_bias).flatten()
        # define final units
        self.I = TT.concatenate((self.ARGS, self.EMB_CONN))
        self.I2Y = theano.shared(value=HE_UNIFORM((self.n_y,
                                                   self.ndim + self.intm_dim)),
                                 name="I2Y")
        self.y_bias = theano.shared(value=HE_UNIFORM((1, self.n_y)),
                                    name="y_bias")
        self._params.extend([self.I2Y, self.y_bias])
        self.Y_pred = TT.nnet.softmax(TT.dot(self.I2Y, self.I).T + self.y_bias)
        # initialize cost and optimization functions
        self.Y_gold = TT.vector(name="Y_gold")
        self._cost = TT.sum((self.Y_pred - self.Y_gold) ** 2)
        self._dev_cost = TT.sum((self.Y_pred - self.Y_gold) ** 2)
        self._pred_class = TT.argmax(self.Y_pred)
        grads = TT.grad(self._cost, wrt=self._params)
        self._init_funcs(grads)
