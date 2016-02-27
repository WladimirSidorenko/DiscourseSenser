#!/usr//bin/env python
# -*- coding: utf-8; mode: python; -*-

"""Package providing modules for sense disambiguation of connectives.

Attributes:
__all__ (list): variables and modules provided for export
__version__ (str): version of this package
constants (module): module containing package constants
scorer (module): module containing CoNLL evaluation scripts
senser (module): module containing methods for sense disambiguation

"""

##################################################################
# Imports
from __future__ import absolute_import, unicode_literals

from dsenser.constants import DFLT_MODEL_PATH, DFLT_MODEL_TYPE, \
    FFNN, LSTM, MJR, SVM, WANG
from dsenser.dsenser import DiscourseSenser

##################################################################
# Variables and Constants
__all__ = ["DiscourseSenser", "DFLT_MODEL_PATH", "DFLT_MODEL_TYPE",
           "FFNN", "LSTM", "MJR", "SVM", "WANG"]
