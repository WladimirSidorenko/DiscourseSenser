#!/usr//bin/env python
# -*- coding: utf-8; mode: python; -*-

"""Package providing modules for XGBoost sense disambiguation.

Attributes:
__all__ (list): variables and modules provided for export
__version__ (str): version of this package
xgboost (module): main module providing XGBoost sense classifier
explicit (module): module providing sense classifier of explicit connectives
implicit (module): module providing sense classifier of implicit connectives

"""

##################################################################
# Imports
from __future__ import absolute_import, unicode_literals

from dsenser.xgboost.xgboost import XGBoostSenser

##################################################################
# Variables and Constants
__all__ = ["XGBoostSenser"]
__version__ = "0.0.1"
