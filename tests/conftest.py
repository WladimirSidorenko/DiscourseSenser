#!/usr/bin/env python
# -*- coding: utf-8; mode: python; -*-

##################################################################
# Imports
from pytest import fixture

import codecs
import json
import os


##################################################################
# Constants
ENCODING = "utf-8"
DIRNAME = os.path.dirname(__file__)
DATA_DIR = os.path.join(DIRNAME, "data")
TEST_BROWN_PATH = os.path.join(DATA_DIR, "brown_cluster_1000.txt")
TRAIN_X_PATH = os.path.join(DATA_DIR, "train_x.json")
with codecs.open(TRAIN_X_PATH, 'r', ENCODING) as ifile:
    _TRAIN_X = json.load(ifile)


##################################################################
# Fixtures
@fixture(scope="module")
def TRAIN_X():
    return _TRAIN_X
