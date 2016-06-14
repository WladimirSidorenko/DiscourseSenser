#!/usr/bin/env python
# -*- coding: utf-8; mode: python; -*-

##################################################################
# Imports
from __future__ import absolute_import

from dsenser.xgboost.xgboostbase import XGBoostBaseSenser

from mock import patch, MagicMock
from pytest import fixture
from unittest import TestCase

import numpy as np
import xgboost
import sklearn

##################################################################
# Constants
N_Y = 5
EPS = 0.499
TRG_CLS = 1
IV = np.zeros((1, 5)) + EPS / N_Y
IV[0, TRG_CLS] = 1. - EPS


##################################################################
# Test Classes
class TestXGBoostBaseSenser(TestCase):
    @fixture(autouse=True)
    def set_ds(self):
        with patch("xgboost.XGBClassifier"):
            with patch("sklearn.pipeline.Pipeline"):
                self.xgb = XGBoostBaseSenser()

    def test_predict_0(self):
        with patch.object(self.xgb._model, "predict_proba",
                          return_value=IV):
            with patch.object(self.xgb, "_clf", MagicMock()):
                with patch.object(self.xgb._clf, "_le", MagicMock()):
                    with patch.object(self.xgb._clf._le, "inverse_transform",
                                      lambda x: x):
                        ov = np.zeros((1, 5))
                        self.xgb._predict({}, ov, 0)
                        assert np.allclose(IV, ov)

    def test_predict_1(self):
        with patch.object(self.xgb._model, "predict_proba",
                          return_value=IV):
            with patch.object(self.xgb, "_clf", None):
                ov = np.zeros((1, 5))
                self.xgb._predict({}, ov, 0)
                assert np.allclose(IV, ov)
