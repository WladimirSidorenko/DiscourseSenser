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
from dsenser.xgboost.xgboostbase import XGBoostBaseSenser, \
    BASE_N_JOBS, BASE_PARAM_GRID


##################################################################
# Classes
class XGBoostImplicitSenser(XGBoostBaseSenser, WangImplicitSenser):
    """Subclass of implicit WangSenser using XGBoost.

    """
    PARAM_GRID = BASE_PARAM_GRID
    N_JOBS = BASE_N_JOBS

    @timeit("Training implicit XGBoost classifier...")
    def train(self, *args, **kwargs):
        super(WangImplicitSenser, self).train(*args, **kwargs)
