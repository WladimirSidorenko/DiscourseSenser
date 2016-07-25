#!/usr/bin/env python
# -*- coding: utf-8; mode: python; -*-

##################################################################
# Imports
from __future__ import absolute_import

from dsenser.nnbase import _norm_vec, _norm_word, NNBaseSenser

from pytest import fixture
from unittest import TestCase
import numpy as np


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
        pass


class TestNNBaseSenser(TestCase):
    @fixture(autouse=True)
    def set_nnbase(self):
        self.nnbs = NNBase()

    def test_init(self):
        assert self.nnbs
