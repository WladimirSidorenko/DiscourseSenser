#!/usr/bin/env python
# -*- coding: utf-8; mode: python; -*-

##################################################################
# Imports
from __future__ import absolute_import, print_function, unicode_literals

from dsenser.wang.wangbase import WangBaseSenser, NFOLDS
import dsenser

from mock import patch, MagicMock, DEFAULT as MOCK_DEFAULT
from pytest import fixture
from unittest import TestCase

import numpy as np
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


class TestWangBase(TestCase):
    @fixture(autouse=True)
    def set_wb(self, TRAIN_X):
        self.TRAIN_X = TRAIN_X
        self.wb = Wangbase()
        self.wb_gs = Wangbase(a_grid_search=True)

    def test_train(self):
        with patch.multiple(self.wb,
                            _extract_features=MOCK_DEFAULT,
                            _model=MOCK_DEFAULT):
            self.wb.train(([], []), None)

    def test_grid_search(self):
        with patch("dsenser.wang.wangbase.GridSearchCV"):
            with patch.multiple(self.wb_gs,
                                _generate_ts=lambda *x: (self.TRAIN_X, Y),
                                _model=MOCK_DEFAULT):
                self.wb_gs.train(([], []), None)

    def test_devset_cv(self):
        y = Y * NFOLDS
        train_N = len(y)
        dev_N = len(Y)
        folds = self.wb_gs._devset_cv(y, dev_N, NFOLDS)
        assert len(folds) == NFOLDS
        dev_idcs = [train_N + i for i in xrange(dev_N)]
        assert np.allclose(folds[0][-1][-dev_N:], dev_idcs)
        assert np.allclose(folds[-1][-1][-dev_N:], dev_idcs)

    def set_free(self):
        self.wb._free()
        assert self.wb < 0
