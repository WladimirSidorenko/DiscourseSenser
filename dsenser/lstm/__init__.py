#!/usr/bin/env python
# -*- coding: utf-8; mode: python; -*-

"""Package providing modules for LSTM sense classification.

Attributes:
  __all__ (list): variables and modules provided for export
  __version__ (str): version of this package
  lstm (module): main module providing LSTMSenser
  lstmbase (module): module providing base abstract LSTMSenser
  explicit (module): module providing explicit LSTMSenser
  implicit (module): module providing implicit LSTMSenser

"""

##################################################################
# Imports
from __future__ import absolute_import, unicode_literals

from dsenser.lstm.lstm import LSTMSenser

##################################################################
# Variables and Constants
__all__ = ["LSTMSenser"]
__version__ = "0.0.1"
