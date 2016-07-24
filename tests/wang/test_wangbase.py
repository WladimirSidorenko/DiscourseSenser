#!/usr/bin/env python
# -*- coding: utf-8; mode: python; -*-

##################################################################
# Imports
from __future__ import absolute_import, print_function, unicode_literals

from test_wang_implicit import REL1, PARSE1

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
N_CLASSES = 5
FEATS = {"f1": 1, "f2": 2}
CLASSES_ = {i: i for i in xrange(N_CLASSES)}


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
        self.TRAIN_X = TRAIN_X

    def test_train_0(self):
        with patch.multiple(self.wb,
                            _extract_features=MOCK_DEFAULT,
                            _model=MOCK_DEFAULT):
            self.wb.train(([], []), None)

    def test_train_1(self):
        j = 2
        ret_train = np.zeros((6, 3, N_CLASSES))
        ret_dev = np.zeros((6, 3, N_CLASSES))
        func_ret = np.zeros((1, N_CLASSES))
        func_ret[0, j] = 1.
        with patch.multiple(self.wb,
                            _gs=True,
                            _generate_ts=lambda *x: (self.TRAIN_X, Y),
                            _extract_features=MagicMock(
                                return_value=FEATS),
                            _model=MOCK_DEFAULT):
            with patch("dsenser.wang.wangbase.GridSearchCV"):
                self.wb._model.decision_function = \
                    MagicMock(return_value=func_ret)
                self.wb._model.classes_ = CLASSES_
                self.wb.train(([(0, REL1)], [PARSE1]),
                              ([(0, REL1)], [PARSE1]),
                              1, 1, ret_train, ret_dev)

    def test_predict(self):
        i = 1
        j = 2
        gold_ret = np.zeros((3, N_CLASSES))
        gold_ret[i, :] = [0.14884758, 0.14884758, 0.40460968,
                          0.14884758, 0.14884758]
        func_ret = np.zeros((1, N_CLASSES))
        func_ret[0, j] = 1.
        real_ret = np.zeros((3, N_CLASSES))
        with patch.object(self.wb, "_extract_features",
                          lambda *x: FEATS):
            with patch.object(self.wb, "_model"):
                self.wb._model.decision_function = \
                    MagicMock(return_value=func_ret)
                self.wb._model.classes_ = CLASSES_
                self.wb.predict(REL1, ((), PARSE1), real_ret, i)
        print("real_ret =", repr(real_ret), file=sys.stderr)
        assert np.sum(real_ret[i, :]) == 1.
        assert np.allclose(real_ret, gold_ret)

    def test_free(self):
        self.wb._free()
        assert self.wb.n_y < 0


class TestWangBaseGridSearch(TestCase):
    @pytest.fixture(autouse=True)
    def set_wb(self, TRAIN_X):
        self.TRAIN_X = TRAIN_X
        self.wb = Wangbase(a_grid_search=True)

    def test_train_0(self):
        with patch("dsenser.wang.wangbase.GridSearchCV"):
            with patch.multiple(self.wb,
                                _generate_ts=lambda *x: (self.TRAIN_X, Y),
                                _model=MOCK_DEFAULT):
                self.wb.train(([], []), None)

    def test_train_1(self):
        rels = [REL1] * NFOLDS
        parses = [PARSE1] * NFOLDS
        with patch("dsenser.wang.wangbase.GridSearchCV"):
            with patch.multiple(self.wb,
                                _generate_ts=lambda *x: (self.TRAIN_X, Y),
                                _model=MOCK_DEFAULT):
                self.wb.train((rels, parses),
                              (rels, parses))

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
                          return_value=FEATS):
            x, y = self.wb._generate_ts(([(0, {SENSE: np.array(xrange(10))}),
                                          (1, {SENSE:
                                               np.array(reversed(xrange(10)))})
                                          ], [[], []]))
        assert x == [(0, FEATS), (1, FEATS)]
        assert y == [9, 0]

    def test_predict_0(self):
        ret = np.zeros((1, 5))
        with patch.object(self.wb, "_model", MagicMock()):
            self.wb._model.decision_function = lambda *x: np.zeros((1, 5))
            self.wb._model.classes_ = {i: i for i in xrange(5)}
            self.wb._predict({}, ret, 0)
        assert np.allclose(ret, np.ones((1, 5)) * 1. / 5.)
