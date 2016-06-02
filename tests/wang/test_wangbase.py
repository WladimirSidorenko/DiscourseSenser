#!/usr/bin/env python
# -*- coding: utf-8; mode: python; -*-

##################################################################
# Imports
from __future__ import absolute_import

from dsenser.wang.wangbase import WangBaseSenser
import dsenser

from mock import patch, MagicMock, DEFAULT as MOCK_DEFAULT
from pytest import fixture
from unittest import TestCase


##################################################################
# Constants


##################################################################
# Test Classes
class Wangbase(WangBaseSenser):
    _model = None

    def _extract_features(self, a_rel, a_parses):
        pass


class TestWangBase(TestCase):
    @fixture(autouse=True)
    def set_wb(self):
        self.wb = Wangbase()

    def test_train(self):
        with patch.multiple(self.wb,
                            _extract_features=MOCK_DEFAULT,
                            _model=MOCK_DEFAULT):
            self.wb.train(([], []), None)

    def set_free(self):
        self.wb._free()
        assert self.wb < 0
