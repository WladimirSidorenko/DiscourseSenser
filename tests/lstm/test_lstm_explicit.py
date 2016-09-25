#!/usr/bin/env python
# -*- coding: utf-8; mode: python; -*-

##################################################################
# Imports
from __future__ import absolute_import

from dsenser.lstm.explicit import LSTMExplicitSenser
import dsenser

from mock import patch
from unittest import TestCase

##################################################################
# Constants


##################################################################
# Test Classes
class TestLSTMExplict(TestCase):
    def test_train(self):
        with patch("dsenser.lstm.lstmbase.LSTMBaseSenser.train",
                   autospec=True):
            lstm = LSTMExplicitSenser()
            lstm.train([], {})
