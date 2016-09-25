#!/usr/bin/env python
# -*- coding: utf-8; mode: python; -*-

##################################################################
# Imports
from __future__ import absolute_import

from dsenser.lstm import LSTMSenser
from dsenser.lstm.lstmbase import LSTMBaseSenser
import dsenser

from mock import patch, MagicMock
from unittest import TestCase

##################################################################
# Constants


##################################################################
# Test Classes
class TestLSTMSenser(TestCase):
    def test_init(self):
        with patch.object(dsenser.nnbase, "Word2Vec",
                          MagicMock(ndim=300)):
            lstm = LSTMSenser()
            assert isinstance(lstm.explicit, LSTMBaseSenser)
            assert isinstance(lstm.implicit, LSTMBaseSenser)
