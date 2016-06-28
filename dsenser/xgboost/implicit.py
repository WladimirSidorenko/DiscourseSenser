#!/usr/bin/env python
# -*- coding: utf-8; mode: python; -*-

"""Module for XGBoost disambiguation of implicit relations.

Attributes:
  XGBoostImplicitSenser (class):
    class that xgboost sense prediction of implicit relations

"""

##################################################################
# Imports
from __future__ import absolute_import, print_function

from dsenser.utils import timeit
from dsenser.wang.implicit import WangImplicitSenser
from dsenser.xgboost.xgboostbase import XGBoostBaseSenser


##################################################################
# Classes
class XGBoostImplicitSenser(XGBoostBaseSenser, WangImplicitSenser):
    """Subclass of implicit WangSenser using XGBoost.

    """
    PARAM_GRID = {"clf__max_depth": [2**i for i in xrange(1, 6)],
                  "clf__n_estimators": [50**i for i in xrange(1, 7)]}

    @timeit("Training implicit XGBoost classifier...")
    def train(self, *args, **kwargs):
        super(WangImplicitSenser, self).train(*args, **kwargs)
