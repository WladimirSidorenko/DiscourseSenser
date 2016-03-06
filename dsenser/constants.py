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
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(DATA_DIR, "models")
ECONN_DIR = os.path.join(DATA_DIR, "econnectives")
DFLT_MODEL_PATH = os.path.join(MODEL_DIR, "pdtb.sense.model")
DFLT_ECONN_PATH = os.path.join(ECONN_DIR, "ExpConn.txt")
DFLT_ECONN_CSTR_PATH = os.path.join(ECONN_DIR, "cstring.txt")

FFNN = 0
LSTM = 1
MJR = 2
SVM = 4
WANG = 8
DFLT_MODEL_TYPE = MJR

# JSON field names
DOC_ID = "DocID"
ALT_LEX = "AltLex"
ARG1 = "Arg1"
ARG2 = "Arg2"
CHAR_SPAN = "CharacterSpanList"
CONNECTIVE = "Connective"
EXPLICIT = "Explicit"
IMPLICIT = "Implicit"
RAW_TEXT = "RawText"
SENSE = "Sense"
SENTENCES = "sentences"
TOK_LIST = "TokenList"
TYPE = "Type"
TOK_OFFS_IDX = 2
TOK_IDX = 0
