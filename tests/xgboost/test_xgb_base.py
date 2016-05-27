#!/usr/bin/env python
# -*- coding: utf-8; mode: python; -*-

##################################################################
# Imports
from __future__ import absolute_import

from dsenser.xgboost.xgboostbase import XGBoostBaseSenser, EPS

from mock import patch
from pytest import fixture
from unittest import TestCase

import numpy as np
import xgboost
import sklearn

##################################################################
# Constants
N_Y = 5
TRG_CLS = 1


##################################################################
# Test Classes
class TestXGBoostBaseSenser(TestCase):
    @fixture(autouse=True)
    def set_ds(self):
        with patch("xgboost.XGBClassifier"):
            with patch("sklearn.pipeline.Pipeline"):
                self.xgb = XGBoostBaseSenser()

    def test_init(self):
        assert self.xgb.one == 1 - EPS

    def test_predict(self):
        with patch.object(self.xgb._model, "predict",
                          return_value=[TRG_CLS]):
            with patch.object(self.xgb, "n_y", N_Y):
                iv = np.zeros((1, 5)) + EPS / N_Y
                iv[0, TRG_CLS] = 1. - EPS
                ov = np.zeros((1, 5))
                self.xgb._predict({}, ov, 0)
                assert np.allclose(iv, ov)
