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
from dsenser.xgboost.xgboostbase import XGBoostBaseSenser, \
    BASE_N_JOBS, BASE_PARAM_GRID


##################################################################
# Classes
class XGBoostExplicitSenser(XGBoostBaseSenser, WangExplicitSenser):
    """Subclass of explicit WangSenser using XGBoost.

    """
    PARAM_GRID = BASE_PARAM_GRID
    N_JOBS = BASE_N_JOBS

    @timeit("Training explicit XGBoost classifier...")
    def train(self, *args, **kwargs):
        super(WangExplicitSenser, self).train(*args, **kwargs)
