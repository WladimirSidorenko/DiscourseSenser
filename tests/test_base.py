#!/usr/bin/env python
# -*- coding: utf-8; mode: python; -*-

##################################################################
# Imports
from __future__ import absolute_import

from dsenser.base import BaseSenser

from mock import Mock, patch
from pytest import fixture
from unittest import TestCase

##################################################################
# Variables and Constants
DS = []


##################################################################
# Test Classes
class TestDiscourseSenser(TestCase):
    @fixture(autouse=True)
    def set_ds(self):
        self.ds = BaseSenser()

    def test_init(self):
        assert self.ds

    def test_train(self):
        with patch.multiple(self.ds, implicit=Mock(),
                            explicit=Mock()):
            self.ds.train(None)

    def test_free(self):
        self.ds._free()

    def test_divide_ds0(self):
        assert self.ds._divide_ds(None) == (([], {}), ([], {}))

    def test_divide_ds1(self):
        assert self.ds._divide_ds(([], {})) == \
            (([], {}), ([], {}))

    def test_normalize_conn0(self):
        assert self.ds._normalize_conn("years before") == "before"

    def test_normalize_conn1(self):
        assert self.ds._normalize_conn("") == ""
