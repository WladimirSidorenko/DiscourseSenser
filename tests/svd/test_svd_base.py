#!/usr/bin/env python
# -*- coding: utf-8; mode: python; -*-

##################################################################
# Imports
from __future__ import absolute_import, unicode_literals, print_function

from dsenser.svd.svdbase import SVDBaseSenser, MIN_DIM
from dsenser.theano_utils import HE_UNIFORM, TT, floatX
import dsenser

from mock import patch, MagicMock
from pytest import fixture
from unittest import TestCase

from theano.tensor.var import TensorVariable
import numpy as np
import theano

##################################################################
# Constants


##################################################################
# Test Classes
class TestSVDBaseSenser(TestCase):
    @fixture(autouse=True)
    def set_svd(self):
        with patch.object(dsenser.nnbase, "Word2Vec",
                          MagicMock(ndim=300)):
            self.svd = SVDBaseSenser(a_w2v=True)
            self.svd.w_i = 50
            self.svd.n_y = 20
            self.svd._init_nn()

    def test_svd_0(self):
        # check that the variables are of the right size
        assert self.svd.intm_dim >= MIN_DIM
        assert isinstance(self.svd.W_INDICES_ARG1,
                          TensorVariable)
        assert self.svd.W_INDICES_ARG1.type.ndim == 1
        assert isinstance(self.svd.W_INDICES_ARG2,
                          TensorVariable)
        assert self.svd.CONN_INDEX.type.ndim == 0

    def test_svd_1(self):
        # compile function that takes preliminary inout and outputs SVD
        get_svd = theano.function([self.svd.EMB_ARG1], self.svd.ARG1,
                                  name="get_svd")
        ret = get_svd(floatX(np.random.randn(20, 30)))
        import sys
        print(ret.size, ret.shape, file=sys.stderr)
        a = ret.dot(ret.T)
        assert np.allclose(a.diagonal(), np.ones(30))
        a -= np.eye(30)
        assert a.max() < 1.e-6
        assert a.min() > -1.e-5
