#!/usr/bin/env python
# -*- coding: utf-8; mode: python; -*-

##################################################################
# Imports
from __future__ import absolute_import

from dsenser.constants import CONNECTIVE, RAW_TEXT, SENSE
from dsenser.major import MajorSenser

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

    def test_ms_train(self):
        irels = [{CONNECTIVE: {RAW_TEXT: "SINCE"},
                  SENSE: ["Contingency.Cause.Reason"]},
                 {CONNECTIVE: {RAW_TEXT: "though"},
                  SENSE: ["Comparison.Contrast"]}]
        train_data = (irels, None)
        dev_data = ([irels[0], ], None)
        n_y = 2
        i = 1
        train_out = np.zeros((2, 2, 2))
        dev_out = np.zeros((1, 2, 2))
        self.ds.train(train_data, dev_data, n_y,
                      i, train_out, dev_out)
        assert self.ds.n_y == 2
        assert np.allclose(self.ds.conn2sense["since"],
                           np.array([1., 0]))

    def test_ms_predict_0(self):
        conn2sense = self.ds.conn2sense
        try:
            stat = np.array(range(1, 11)) / 55.
            conn_stat = {"since": stat}
            self.ds.conn2sense = conn_stat
            ret = np.zeros((3, 10))
            self.ds.predict({CONNECTIVE: {RAW_TEXT: "SiNCe"}},
                            None, ret, 1)
            assert not np.any(ret[0])
            assert not np.any(ret[-1])
            assert np.allclose(ret[1], stat)
        finally:
            self.ds.conn2sense = conn2sense

    def test_ms_predict_1(self):
        dflt_sense = self.ds.dflt_sense
        conn2sense = self.ds.conn2sense
        try:
            stat = np.array(range(1, 11)) / 55.
            self.ds.dflt_sense = stat
            self.ds.conn2sense = {}
            ret = np.zeros((3, 10))
            self.ds.predict({CONNECTIVE: {RAW_TEXT: "because"}},
                            None, ret, 1)
            assert not np.any(ret[0])
            assert not np.any(ret[-1])
            assert np.allclose(ret[1], stat)
        finally:
            self.ds.dflt_sense = dflt_sense
            self.ds.conn2sense = conn2sense

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
