#!/usr/bin/env python
# -*- coding: utf-8; mode: python; -*-

##################################################################
# Imports
from __future__ import absolute_import

from dsenser import DiscourseSenser

from pytest import fixture
from unittest import TestCase

##################################################################
# Variables and Constants


##################################################################
# Test Classes
class TestDiscourseSenser(TestCase):
    @fixture(autouse=True)
    def set_ds(self):
        self.ds = DiscourseSenser()

    def test_init(self):
        assert self.ds
