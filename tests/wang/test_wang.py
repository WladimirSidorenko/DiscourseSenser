#!/usr/bin/env python
# -*- coding: utf-8; mode: python; -*-

##################################################################
# Imports
from __future__ import absolute_import

from unittest import TestCase
from dsenser.wang import WangSenser

##################################################################
# Constants


##################################################################
# Test Classes
class TestWangSenser(TestCase):
    def test_init(self):
        wang = WangSenser()
        assert wang.implicit is not None
        assert wang.explicit is not None
        assert wang.n_y < 0
