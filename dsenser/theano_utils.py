#!/usr//bin/env python
# -*- coding: utf-8; mode: python; -*-

"""Module providing common heano utils.

Attributes:
floatX (method): force float type expected by theano
rmsprop (method): separate training set into explicit and implicit instances

"""

##################################################################
# Imports
from __future__ import absolute_import, print_function

from lasagne.init import HeNormal, HeUniform, Orthogonal
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano import config, tensor as TT
import numpy as np
import theano


##################################################################
# Methods
def floatX(a_data, a_dtype=config.floatX):
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
    x (list):
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
    zipped_grads = [theano.shared(p.get_value() * floatX(0.))
                    for p in tparams]
    running_grads = [theano.shared(p.get_value() * floatX(0.))
                     for p in tparams]
    running_grads2 = [theano.shared(p.get_value() * floatX(0.))
                      for p in tparams]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(x + [y], cost,
                                    updates=zgup + rgup + rg2up,
                                    name='rmsprop_f_grad_shared')
    updir = [theano.shared(p.get_value() * floatX(0.))
             for p in tparams]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / TT.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(tparams, updir_new)]
    f_update = theano.function([], [], updates=updir_new + param_up,
                               on_unused_input='ignore',
                               name='rmsprop_f_update')
    params = [zipped_grads, running_grads, running_grads2, updir]
    return (f_grad_shared, f_update, params)


##################################################################
# Variables and Constants
MAX_ITERS = 3  # 00  # 450
CONV_EPS = 1e-5
DFLT_VDIM = 100

_HE_NORMAL = HeNormal()
HE_NORMAL = lambda x: floatX(_HE_NORMAL.sample(x))

_HE_UNIFORM = HeUniform()
HE_UNIFORM = lambda x: floatX(_HE_UNIFORM.sample(x))

_HE_UNIFORM_RELU = HeUniform(gain=np.sqrt(2))
HE_UNIFORM_RELU = lambda x: floatX(_HE_UNIFORM_RELU.sample(x))

_RELU_ALPHA = 0.
_HE_UNIFORM_LEAKY_RELU = HeUniform(
    gain=np.sqrt(2. / (1 + (_RELU_ALPHA or 1e-6)**2)))
HE_UNIFORM_LEAKY_RELU = lambda x: \
    floatX(_HE_UNIFORM_LEAKY_RELU.sample(x))

_ORTHOGONAL = Orthogonal()
ORTHOGONAL = lambda x: floatX(_ORTHOGONAL.sample(x))

TRNG = RandomStreams()
