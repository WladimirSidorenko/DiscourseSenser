#!/usr/bin/env python
# -*- coding: utf-8; mode: python; -*-

##################################################################
# Imports
from __future__ import absolute_import

from dsenser.nnbase import _norm_vec, _norm_word, NNBaseSenser

from pytest import fixture
from mock import patch
from unittest import TestCase
import numpy as np
import pytest


##################################################################
# Test Methods
def test_norm_vec():
    a = np.ones(10)
    b = _norm_vec(a)
    assert np.sqrt(np.sum(b**2)) == 1.


def test_norm_word():
    assert _norm_word("124345") == "1"
    assert _norm_word("ABCDEF") == "abcdef"


##################################################################
# Test Classes
class NNBase(NNBaseSenser):
    def _init_nn(self):
        """Initialize neural network.

        """
        super(NNBase, self)._init_nn()


class TestNNBaseSenser(TestCase):
    @fixture(autouse=True)
    def set_nnbase(self):
        self.nnbs = NNBase()

    def test_init(self):
        assert self.nnbs

    def test_init_nn(self):
        with pytest.raises(NotImplementedError):
            self.nnbs._init_nn()

    def test_free(self):
        self.nnbs._free()
        assert self.nnbs.n_y < 0
        assert self.nnbs._w_stat is None
        assert len(self.nnbs._params) == 0

    def test_init_w_emb(self):
        with patch.multiple(self.nnbs, _params=[], ndim=3):
            self.nnbs._init_w_emb()
            assert len(self.nnbs._params) == 1
