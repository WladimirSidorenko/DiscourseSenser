#!/usr/bin/env python
# -*- coding: utf-8; mode: python; -*-

##################################################################
# Imports
from __future__ import absolute_import

from dsenser.svd import SVDSenser
from dsenser.svd.svdbase import SVDBaseSenser
import dsenser

from mock import patch, MagicMock
from pytest import fixture
from unittest import TestCase

##################################################################
# Constants


##################################################################
# Test Classes
class TestSVDSenser(TestCase):
    @fixture(autouse=True)
    def test_init(self):
        with patch.object(dsenser.nnbase, "Word2Vec",
                          MagicMock(ndim=300)):
            svd = SVDSenser()
            assert isinstance(svd.explicit, SVDBaseSenser)
            assert isinstance(svd.implicit, SVDBaseSenser)
