#!/usr/bin/env python
# -*- coding: utf-8; mode: python; -*-

"""Module for XGBoost disambiguation of explicit relations.

Attributes:
  XGBoostExplicitSenser (class):
    class that xgboost sense prediction of explicit relations

"""

##################################################################
# Imports
from __future__ import absolute_import, print_function

from dsenser.utils import timeit
from dsenser.wang.explicit import WangExplicitSenser
from dsenser.xgboost.xgboostbase import XGBoostBaseSenser


##################################################################
# Classes
class XGBoostExplicitSenser(XGBoostBaseSenser, WangExplicitSenser):
    """Subclass of explicit WangSenser using XGBoost.

    """
    PARAM_GRID = {"clf__max_depth": [2**i for i in xrange(1, 6)],
                  "clf__n_estimators": [50**i for i in xrange(1, 7)]}

    @timeit("Training explicit XGBoost classifier...")
    def train(self, *args, **kwargs):
        super(WangExplicitSenser, self).train(*args, **kwargs)
