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


##################################################################
# Test Classes
class TestXGBoostBaseSenser(TestCase):
    @fixture(autouse=True)
    def set_ds(self):
        with patch("xgboost.XGBClassifier"):
            with patch("sklearn.pipeline.Pipeline"):
                self.xgb = XGBoostBaseSenser()

    def test_predict(self):
        iv = np.zeros((1, 5)) + EPS / N_Y
        iv[0, TRG_CLS] = 1. - EPS
        with patch.object(self.xgb._model, "predict_proba",
                          return_value=iv):
            with patch.object(self.xgb, "_clf", MagicMock()):
                with patch.object(self.xgb._clf, "_le", MagicMock()):
                    with patch.object(self.xgb._clf._le, "inverse_transform",
                                      lambda x: x):
                        ov = np.zeros((1, 5))
                        self.xgb._predict({}, ov, 0)
                        assert np.allclose(iv, ov)
