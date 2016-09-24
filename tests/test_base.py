#!/usr/bin/env python
# -*- coding: utf-8; mode: python; -*-

##################################################################
# Imports
from __future__ import absolute_import

from dsenser.base import BaseSenser
from dsenser.constants import CONNECTIVE, TOK_LIST

from mock import Mock, patch
from pytest import fixture
from unittest import TestCase

import numpy as np

##################################################################
# Variables and Constants
PRED_IDX = 0
EXPL_REL = {CONNECTIVE: {TOK_LIST: [1, 2, 3]}}
IMPL_REL = {CONNECTIVE: {TOK_LIST: []}}


##################################################################
# Methods
def _mocked_predict(a_rel, a_data, a_ret, a_i):
    """Mocked version of the prediction method.

    Sets `a_ret[a_i][0]` to `1`.

    Args:
      a_rel (dict):
        discourse relation whose sense should be predicted
      a_data (2-tuple(dict, dict)):
        list of input JSON data
      a_ret (np.array):
        output prediction vector
      a_i (int):
        row index in the output vector

    Returns:
      void:

    Note:
      modifies ``a_ret[a_i]`` in place

    """
    a_ret[a_i] *= 0
    a_ret[a_i][PRED_IDX] = 1.


##################################################################
# Test Classes
class TestDiscourseSenser(TestCase):
    @fixture(autouse=True)
    def set_ds(self):
        self.ds = BaseSenser()

    def test_init(self):
        assert self.ds

    def test_train(self):
        with patch.multiple(self.ds, implicit=Mock(),
                            explicit=Mock()):
            self.ds.train(None)

    def test_free(self):
        self.ds._free()

    def test_predict_explicit(self):

        ret = np.zeros((5, 10))
        idx = 4
        with patch.object(self.ds, "explicit"):
            self.ds.explicit.predict = _mocked_predict
            self.ds.predict(EXPL_REL, {}, ret, idx)
        assert ret[idx, PRED_IDX] != 0.
        ret[idx, PRED_IDX] = 0.
        assert not np.any(ret)

    def test_predict_implicit(self):

        ret = np.zeros((5, 10))
        idx = 4
        with patch.object(self.ds, "implicit"):
            self.ds.implicit.predict = _mocked_predict
            self.ds.predict(IMPL_REL, {}, ret, idx)
        assert ret[idx, PRED_IDX] != 0.
        ret[idx, PRED_IDX] = 0.
        assert not np.any(ret)

    def test_batch_predict(self):
        ret = np.ones((5, 10))
        with patch.object(self.ds, "explicit"):
            self.ds.explicit.predict = _mocked_predict
            self.ds.batch_predict([EXPL_REL] * 5, {}, ret)
        assert np.all(ret[:, PRED_IDX])
        ret[:, PRED_IDX] = 0.
        assert not np.any(ret)

    def test_divide_ds0(self):
        assert self.ds._divide_ds(None) == (([], {}), ([], {}))

    def test_divide_ds1(self):
        assert self.ds._divide_ds(([], {})) == \
            (([], {}), ([], {}))

    def test_divide_ds2(self):
        assert self.ds._divide_ds(([EXPL_REL, IMPL_REL], {})) == \
            (([(0, EXPL_REL)], {}), ([(1, IMPL_REL)], {}))

    def test_normalize_conn0(self):
        assert self.ds._normalize_conn("years before") == "before"

    def test_normalize_conn1(self):
        assert self.ds._normalize_conn("") == ""
