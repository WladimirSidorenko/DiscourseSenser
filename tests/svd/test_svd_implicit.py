#!/usr/bin/env python
# -*- coding: utf-8; mode: python; -*-

##################################################################
# Imports
from __future__ import absolute_import

from dsenser.svd.implicit import SVDImplicitSenser
import dsenser

from mock import patch
from unittest import TestCase

##################################################################
# Constants


##################################################################
# Test Classes
class TestSVDImplict(TestCase):
    def test_train(self):
        with patch("dsenser.svd.svdbase.SVDBaseSenser.train",
                   autospec=True):
            svd = SVDImplicitSenser()
            svd.train([], {})
