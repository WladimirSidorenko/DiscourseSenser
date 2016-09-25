#!/usr/bin/env python
# -*- coding: utf-8; mode: python; -*-

##################################################################
# Imports
from __future__ import absolute_import

from dsenser.lstm.implicit import LSTMImplicitSenser
import dsenser

from mock import patch
from unittest import TestCase

##################################################################
# Constants


##################################################################
# Test Classes
class TestLSTMImplict(TestCase):
    def test_train(self):
        with patch("dsenser.lstm.lstmbase.LSTMBaseSenser.train",
                   autospec=True):
            lstm = LSTMImplicitSenser()
            lstm.train([], {})
