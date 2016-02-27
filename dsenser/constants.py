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
DFLT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "data", "models",
                               "pdtb.sense.model")

FFNN = 0
LSTM = 1
MJR = 2
SVM = 4
WANG = 8
DFLT_MODEL_TYPE = MJR

# JSON field names
ARG1 = "Arg1"
ARG2 = "Arg2"
CHAR_SPAN = "CharacterSpanList"
CONNECTIVE = "Connective"
EXPLICIT = "Explicit"
IMPLICIT = "Implicit"
RAW_TEXT = "RawText"
SENSE = "Sense"
TOK_LIST = "TokenList"
TYPE = "Type"
TOK_OFFS_IDX = 2
