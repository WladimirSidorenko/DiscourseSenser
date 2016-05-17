#!/usr/bin/env python
# -*- coding: utf-8; mode: python; -*-

"""Module containing constants for dsenser package.

Attributes:
  ALT_LEX (str): symbolic name of the `alternative lexicalization` relation
    type
  ARG1 (str): symbolic name of the field containing the 1-st argument of the
    input relations in the JSON data
  ARG2 (str): symbolic name of the field containing the 2-nd argument of the
    input relations in the JSON data
  CHAR_SPAN (str): symbolic name of the field containing the character
   span list in the input JSON data
  COB (str): symbolic name of the field containing the beggining of the
   character offset in the input JSON data
  COE (str):symbolic name of the field containing the end of the
   character offset in the input JSON data
  CONNECTIVE (str): symbolic name of the field containing the connectives
    in the input JSON data
  DATA_DIR (str): path to the directory with auxliary data files
  DEPS (str): symbolic name of the field containing grammar dependencies
    in the input JSON data
  DFLT_BROWN_PATH (str): path to the file containing Brown clusters
  DFLT_ECONN_PATH (str): path to the file containing explicit connectives
  DFLT_INQUIRER_PATH (str):  path to the file containing General Inquirer
    lexicon
  DFLT_LCSI_PATH (str): path to the file with LCSInfomerge
  DFLT_MODEL_PATH (str): path to the main default model file
  DFLT_MODEL_TYPE (int): default type of the model to be trained
  DFLT_MPQA_PATH (str): path to the MPQA lexicon
  DFLT_W2V_PATH (str): path to Google word2vec file
  DIRNAME (str): path to the package directory
  DOC_ID (str): symbolic name of the field containing document id in the
    input JSON data
  ECONN_DIR (str): path to the directory containing file with the explcit
    connectives
  ENCODING (str): default encoding used for input and output files
  EXPLICIT (str): symbolic name of the `explicit` relation type
  IMPLICIT (str): symbolic name of the `implicit` relation type
  LSTM (int): integer flag for LSTM training mode
  MJR (int): integer flag for MajorSenser training mode
  MODEL_DIR (str): directory containing the main default model file
  PARSE_TREE (str): symbolic name of the field containing syntactic
    parse tree in the input JSON data
  POS (str): symbolic name of the field containing part-of-speech tag
    parse tree in the input JSON data
  RAW_TEXT (str): symbolic name of the field containing the raw text
    of connectives, arguments, and sentences in the input JSON data
  SENSE (str): symbolic name of the field containing the sense of
    a relation in the input JSON data
  SENTENCES (str): symbolic name of the field containing the information
    about the input senteces in the input JSON data
  SHORT2FULL (dict): mapping from partial to full sense names
  SNT_ID (int): index of the sentence id in the list containing token
    information
  SSNT_ID (str): symbolic name of the field containing the information
    about the sentece ids in the input JSON data
  SVD (int): integer flag for SVD training mode
  TOK_ID (int): index of the sentence token id in the list containing token
    information
  TOK_IDX (int): index of the document token id in the list containing token
    information
  TOK_LIST (str): symbolic name of the field containing the list
    of tokens in the input JSON data
  TOK_OFFS_IDX (int): index of the token offset information in the list
    containing the token information
  TYPE (str): symbolic name of the field containing the information
    about the type of an input relation in the input JSON data
  WANG (int): integer flag for Wang training mode
  WORDS (str): symbolic name of the field containing the information
    about the words of an input relation in the input JSON data
  XGBOOST (int): integer flag for XGBoost training mode

"""

##################################################################
# Imports
from __future__ import absolute_import

import os

##################################################################
# Variables and Constants
ENCODING = "utf-8"

DIRNAME = os.path.dirname(__file__)
DATA_DIR = os.path.join(DIRNAME, "data")

MODEL_DIR = os.path.join(DATA_DIR, "models")
DFLT_MODEL_PATH = os.path.join(MODEL_DIR, "pdtb.sense.model")

ECONN_DIR = os.path.join(DATA_DIR, "econnectives")
DFLT_ECONN_PATH = os.path.join(ECONN_DIR, "ExpConn.txt")
DFLT_BROWN_PATH = os.path.join(DATA_DIR, "brown_cluster_1000.txt")
DFLT_INQUIRER_PATH = os.path.join(DATA_DIR, "inquirer_word")
DFLT_LCSI_PATH = os.path.join(DATA_DIR, "LCSInfomerge.txt")
DFLT_MPQA_PATH = os.path.join(DATA_DIR, "MPQA_Subjectivity_Lexicon.txt")
DFLT_W2V_PATH = os.path.join(DATA_DIR, "GoogleNews-vectors-negative300.bin")

SVD = 1
LSTM = 2
MJR = 4
WANG = 8
XGBOOST = 16
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
SSNT_ID = "SentenceId"
TOK_IDX = 0
TOK_LIST = "TokenList"
TOK_OFFS_IDX = 2
TYPE = "Type"
WORDS = "words"
COB = "CharacterOffsetBegin"
COE = "CharacterOffsetEnd"

SNT_ID = 3
TOK_ID = 4
PARSE_TREE = "parsetree"

SHORT2FULL = {"Expansion": "Expansion.Conjunction",
              "Comparison": "Comparison.Contrast",
              "Comparison.COntrast": "Comparison.Contrast"}
