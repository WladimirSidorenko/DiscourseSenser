#!/usr/bin/env python
# -*- coding: utf-8; mode: python; -*-

"""Package providing modules for sense disambiguation of connectives.

Attributes:
  __all__ (list): variables and modules provided for export
  __version__ (str): version of this package
  constants (module): module containing package constants
  dsenser (module): module providing the main interface class
                 :class:`dsenser.DiscourseSenser` for wrapping
                 classifiers into ensembles.
  lstm (module): module providing the class
                 :class:`dsenser.lstm.LSTMSenser` for
                 LSTM-based sense prediction
  major (module): module providing the class
                  :class:`dsenser.major.MajorSenser` for majority class
                  prediction
  svd (module): module providing the class
                :class:`dsenser.svd.SVDSenser` for SVD-based sense prediction
  wang (module): module providing the class
                 :class:`dsenser.wang.WangSenser` for Wang classification
                 using LinearSVC
  xgboost (module): module providing the class
                    :class:`dsenser.xgboost.XGBoostSenser`
                    for Wang classification using XGBoost forrests
  scorer (module): module containing CoNLL evaluation scripts


"""

##################################################################
# Imports
from __future__ import absolute_import, unicode_literals

# neural net sensers are not imported here due to slow loading of theano
from dsenser.constants import DFLT_MODEL_PATH, DFLT_MODEL_TYPE, \
    SVD, LSTM, MJR, WANG, XGBOOST
from dsenser.dsenser import DiscourseSenser
from dsenser.major import MajorSenser
from dsenser.wang import WangSenser

##################################################################
# Variables and Constants
__all__ = ["DiscourseSenser", "MajorSenser", "WangSenser",
           "DFLT_MODEL_PATH", "DFLT_MODEL_TYPE",
           "SVD", "LSTM", "MJR", "WANG", "XGBOOST"]
__version__ = "0.0.3"
