#!/usr/bin/env python
# -*- coding: utf-8; mode: python; -*-

##################################################################
# Imports
from __future__ import absolute_import

from dsenser.word2vec import Word2Vec

from mock import patch, MagicMock
from pytest import fixture
from unittest import TestCase

import pytest

##################################################################
# Variables and Constants
W2V = MagicMock()
DFLT_VEC_SIZE = 300


##################################################################
# Test Classes
class TestWord2Vec(TestCase):
    @fixture(autouse=True)
    def set_w2v(self):
        self.w2v = Word2Vec

    def test_contains_0(self):
        assert not "table" in self.w2v

    def test_contains_1(self):
        with patch.object(self.w2v, "_cache", {"tables": 1}):
            assert "tables" in self.w2v

    def test_contains_2(self):
        with patch.object(self.w2v, "_w2v", {"tablez": 1}):
            assert "tablez" in self.w2v
            assert self.w2v["tablez"] == 1

    def test_getitem_0(self):
        with patch.object(self.w2v, "_cache", {"tablex": 0}):
            assert self.w2v["tablex"] == 0

    def test_getitem_1(self):
        with patch.object(self.w2v, "_w2v", {"tabley": 1}):
            assert self.w2v["tabley"] == 1
            assert "tabley" in self.w2v._cache

    def test_getitem_2(self):
        with pytest.raises(KeyError):
            self.w2v["gable"]

    def test_load(self):
        with patch.object(self.w2v, "_w2v"):
            self.w2v._w2v.load = MagicMock()
            self.w2v._w2v.resource = MagicMock()
            self.w2v._w2v.resource.vector_size = DFLT_VEC_SIZE
            self.w2v.load()
            self.w2v._w2v.load.assert_called_once_with()
            assert self.w2v.ndim == DFLT_VEC_SIZE

    def test_unload(self):
        with patch.object(self.w2v, "_cache"):
            with patch.object(self.w2v, "_w2v"):
                self.w2v._cache.clear = MagicMock()
                self.w2v._w2v._unload = MagicMock()
                self.w2v.unload()
                self.w2v._cache.clear.assert_called_once_with()
