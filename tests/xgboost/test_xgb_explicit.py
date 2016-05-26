#!/usr/bin/env python
# -*- coding: utf-8; mode: python; -*-

##################################################################
# Imports
from __future__ import absolute_import

import dsenser
from dsenser.xgboost.explicit import XGBoostExplicitSenser

from mock import patch
from unittest import TestCase


##################################################################
# Constants


##################################################################
# Test Classes
class TestXGBoostExplict(TestCase):
    def test_train(self):
        with patch("dsenser.wang.wangbase.WangBaseSenser.train",
                   autospec=True):
            xgb = XGBoostExplicitSenser()
            xgb.train([], {})
