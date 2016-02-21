#!/usr//bin/env python
# -*- coding: utf-8; mode: python; -*-

"""Module containing constants for dsenser package.

Attributes:

"""

##################################################################
# Imports
from __future__ import absolute_import

import os

##################################################################
# Variables and Constants
DFLT_MODEL = os.path.join(os.path.dirname(__name__), "data", "models",
                          "pdtb.sense.model")

FFNN = "FFNN"
LSTM = "LSTM"
SVM = "SVM"
