#!/usr/bin/env python
# -*- coding: utf-8; mode: python; -*-

##################################################################
# Imports
from __future__ import absolute_import, print_function, unicode_literals

from dsenser.constants import SENSE
from dsenser.wang.wangbase import WangBaseSenser, NFOLDS
import dsenser

from mock import patch, MagicMock, DEFAULT as MOCK_DEFAULT
from unittest import TestCase

import numpy as np
import pytest
import sys


##################################################################
# Constants
Y = [0, 2, 2, 5, 7, 0]


##################################################################
# Test Classes
class Wangbase(WangBaseSenser):
    _model = None

    def _extract_features(self, a_rel, a_parses):
        pass


class AbstractWangbase(WangBaseSenser):
    _model = None

    def _extract_features(self, a_rel, a_parses):
        super(AbstractWangbase, self)._extract_features(a_rel, a_parses)


class TestWangBaseSenser(TestCase):
    def test_abstract(self):
        with pytest.raises(TypeError):
            WangBaseSenser()

    def test_abstract(self):
        awb = AbstractWangbase()
        with pytest.raises(NotImplementedError):
            awb._extract_features([], [])


class TestWangBase(TestCase):
    @pytest.fixture(autouse=True)
    def set_wb(self, TRAIN_X):
        self.wb = Wangbase()

    def test_train(self):
        with patch.multiple(self.wb,
                            _extract_features=MOCK_DEFAULT,
                            _model=MOCK_DEFAULT):
            self.wb.train(([], []), None)

    def test_free(self):
        self.wb._free()
        assert self.wb.n_y < 0


class TestWangBaseGridSearch(TestCase):
    @pytest.fixture(autouse=True)
    def set_wb(self, TRAIN_X):
        self.TRAIN_X = TRAIN_X
        self.wb = Wangbase(a_grid_search=True)

    def test_train(self):
        with patch("dsenser.wang.wangbase.GridSearchCV"):
            with patch.multiple(self.wb,
                                _generate_ts=lambda *x: (self.TRAIN_X, Y),
                                _model=MOCK_DEFAULT):
                self.wb.train(([], []), None)

    def test_devset_cv(self):
        y = Y * NFOLDS
        train_N = len(y)
        dev_N = len(Y)
        folds = self.wb._devset_cv(y, dev_N, NFOLDS)
        assert len(folds) == NFOLDS
        dev_idcs = [train_N + i for i in xrange(dev_N)]
        assert np.allclose(folds[0][-1][-dev_N:], dev_idcs)
        assert np.allclose(folds[-1][-1][-dev_N:], dev_idcs)

    def test_generate_ts_0(self):
        x, y = self.wb._generate_ts(None)
        assert x == []
        assert y == []

    def test_generate_ts_1(self):
        with patch.object(self.wb, "_extract_features", return_value=None):
            x, y = self.wb._generate_ts(([(0, []), (1, [])], [[], []]))
        assert x == []
        assert y == []

    def test_generate_ts_2(self):
        with patch.object(self.wb, "_extract_features",
                          return_value={"f1": 1, "f2": 2}):
            x, y = self.wb._generate_ts(([(0, {SENSE: np.array(xrange(10))}),
                                          (1, {SENSE:
                                               np.array(reversed(xrange(10)))})
                                          ], [[], []]))
        assert x == [(0, {"f1": 1, "f2": 2}), (1, {"f1": 1, "f2": 2})]
        assert y == [9, 0]

    def test_predict_0(self):
        ret = np.zeros((1, 5))
        with patch.object(self.wb, "_model", MagicMock()):
            self.wb._model.decision_function = lambda *x: np.zeros((1, 5))
            self.wb._model.classes_ = {i: i for i in xrange(5)}
            self.wb._predict({}, ret, 0)
        assert np.allclose(ret, np.zeros((1, 5)))
