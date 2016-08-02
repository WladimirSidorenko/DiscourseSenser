#!/usr/bin/env python
# -*- coding: utf-8; mode: python; -*-

##################################################################
# Imports
from __future__ import absolute_import

from dsenser.theano_utils import floatX, rmsprop

from copy import deepcopy
from theano import config, tensor as TT

import numpy as np
import theano


##################################################################
# Constants
N = 10
MAX_I = 7


##################################################################
# Methods
def test_floatX_0():
    scalar = floatX(0)
    assert scalar.dtype == config.floatX
    assert isinstance(scalar, np.ndarray)


def test_floatX_1():
    scalar = floatX(range(5))
    assert scalar.dtype == config.floatX
    assert isinstance(scalar, np.ndarray)


def test_rmsprop_0():
    # input
    x = TT.vector(name='x')
    B = theano.shared(floatX(np.ones((3, 5))), name='B')
    c = theano.shared(floatX(np.ones(3)), name='c')
    params = [B, c]
    # output
    y_pred = TT.nnet.softmax(TT.dot(B, x.T).T + c)
    y_gold = TT.vector(name="y_gold")
    # cost and grads
    cost = TT.sum((y_pred - y_gold)**2)
    grads = TT.grad(cost, wrt=params)
    # funcs
    cost_func, update_func, rms_params = rmsprop(params, grads,
                                                 [x], y_gold, cost)
    # check return values
    assert len(rms_params) == 4
    assert isinstance(rms_params[0][0], TT.sharedvar.TensorSharedVariable)
    assert not np.any(rms_params[0][0].get_value())
    # check convergence
    X = [floatX(np.random.rand(5)) for _ in xrange(N)]
    Y = [floatX(np.random.rand(3)) for _ in xrange(N)]
    icost = init_cost = end_cost = 0.
    for i in xrange(MAX_I):
        icost = 0.
        for x, y in zip(X, Y):
            icost += cost_func(x, y)
            update_func()
        if i == 0:
            init_cost = icost
        elif i == MAX_I - 1:
            end_cost = icost
    assert end_cost < init_cost
