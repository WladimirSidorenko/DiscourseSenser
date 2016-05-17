#!/usr/bin/env python
# -*- coding: utf-8; mode: python; -*-

##################################################################
# Imports
from __future__ import absolute_import

from dsenser.major import MajorSenser

from mock import Mock, patch
from pytest import fixture
from unittest import TestCase

import numpy as np


##################################################################
# Test Classes
class TestMajorSenser(TestCase):
    @fixture(autouse=True)
    def set_ds(self):
        self.ds = MajorSenser()

    def test_ms(self):
        assert isinstance(self.ds, MajorSenser)
        assert self.ds.dflt_sense is None
        assert len(self.ds.conn2sense) == 0
        assert self.ds.n_y < 0

    def test_ms_get_sense_stat_0(self):
        n_y = self.ds.n_y
        try:
            self.ds.n_y = 2
            stat = {1: 3, 0: 5}
            assert np.allclose(self.ds._get_sense_stat(stat),
                               [5./8., 3./8.])
        finally:
            self.ds.n_y = n_y

    def test_ms_get_sense_stat_1(self):
        self.ds._get_sense_stat({})

    def test_ms_get_sense_stat_2(self):
        n_y = self.ds.n_y
        try:
            self.ds.n_y = 2
            assert np.allclose(self.ds._get_sense_stat({}),
                               np.zeros(2))
        finally:
            self.ds.n_y = n_y

    def test_ms_free(self):
        self.ds._free()
        assert not hasattr(self.ds, "dflt_sense")
        assert not hasattr(self.ds, "conn2sense")
        assert self.ds.n_y < 0
