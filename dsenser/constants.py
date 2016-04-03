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
ENCODING = "utf-8"

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

MODEL_DIR = os.path.join(DATA_DIR, "models")
DFLT_MODEL_PATH = os.path.join(MODEL_DIR, "pdtb.sense.model")

ECONN_DIR = os.path.join(DATA_DIR, "econnectives")
DFLT_ECONN_PATH = os.path.join(ECONN_DIR, "ExpConn.txt")
DFLT_BROWN_PATH = os.path.join(DATA_DIR, "brown_cluster_1000.txt")
DFLT_INQUIRER_PATH = os.path.join(DATA_DIR, "inquirer_word")
DFLT_LCSI_PATH = os.path.join(DATA_DIR, "LCSInfomerge.txt")
DFLT_MPQA_PATH = os.path.join(DATA_DIR, "MPQA_Subjectivity_Lexicon.txt")

FFNN = 0
LSTM = 1
MJR = 2
SVM = 4
WANG = 8
DFLT_MODEL_TYPE = WANG

# JSON field names
ALT_LEX = "AltLex"
ARG1 = "Arg1"
ARG2 = "Arg2"
CHAR_SPAN = "CharacterSpanList"
CONNECTIVE = "Connective"
DEPS = "dependencies"
DOC_ID = "DocID"
EXPLICIT = "Explicit"
IMPLICIT = "Implicit"
POS = "PartOfSpeech"
RAW_TEXT = "RawText"
SENSE = "Sense"
SENTENCES = "sentences"
TOK_IDX = 0
TOK_LIST = "TokenList"
TOK_OFFS_IDX = 2
TYPE = "Type"
WORDS = "words"

SNT_ID = 3
TOK_ID = 4
PARSE_TREE = "parsetree"
