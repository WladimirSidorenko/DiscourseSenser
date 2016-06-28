#!/usr/bin/env python
# -*- coding: utf-8; mode: python; -*-

##################################################################
# Imports
from __future__ import absolute_import

from unittest import TestCase
from dsenser.xgboost import XGBoostSenser

##################################################################
# Constants


##################################################################
# Test Classes
class TestXGBoostSenser(TestCase):
    def test_init(self):
        xgb = XGBoostSenser()
        assert xgb.implicit is not None
        assert xgb.explicit is not None
