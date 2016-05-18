#!/usr/bin/env python
# -*- coding: utf-8; mode: python; -*-

##################################################################
# Imports
from __future__ import absolute_import

from dsenser.resources import BROWN_CLUSTERS, CONNS, DFLT_W2V_PATH, \
    INQUIRER, LCSI, MPQA, STEMMED_INQUIRER, W2V, conn2str

from unittest import TestCase
from mock import patch

import gensim


##################################################################
# Constants
ON_THE_CONTRARY = (("on", "the", "contrary"),)
NEITHER_NOT = (("neither", ), ("nor",))
EMPTY = ((), )


##################################################################
# Test Classes
class TestResources(TestCase):
    def test_BROWN_CLUSTERS(self):
        assert BROWN_CLUSTERS["jasper"] == "1100011110"
        assert BROWN_CLUSTERS["un"] == "1100011110|1110110010|1011010110"
        assert "" not in BROWN_CLUSTERS

    def test_CONNS(self):
        assert ON_THE_CONTRARY in CONNS
        assert NEITHER_NOT in CONNS
        assert EMPTY not in CONNS

    def test_INQUIRER(self):
        assert "" not in INQUIRER
        assert INQUIRER["won"] == [False, False, False, False, True,
                                   False, True, False, False, False,
                                   False, False, False, False, True,
                                   False, False, False, False, False,
                                   False, False, False, False, True,
                                   False, False, False, False, False,
                                   False, False, False, False, False,
                                   False, False, False, False, False,
                                   False, False, False, False, False]
        assert INQUIRER["won"] == STEMMED_INQUIRER["won"]

    def test_LCSI(self):
        assert LCSI["unionize"] == set(["45.4.a", "45.4.b", "45.4.c"])
        assert LCSI["confer"] == set(["37.6.a", "37.6.b"])

    def test_MPQA(self):
        assert MPQA["zealously"] == ("negative", "strongsubj", "anypos")

    def test_conn2str(self):
        assert conn2str(ON_THE_CONTRARY) == "on_the_contrary"
        assert conn2str(NEITHER_NOT) == "neither_nor"
        assert conn2str(EMPTY) == ""

    def test_w2v(self):
        with patch.object(gensim.models.word2vec.Word2Vec,
                          "load_word2vec_format") as mock_method:
            W2V["zzz"]
        mock_method.assert_any_call(DFLT_W2V_PATH, binary=True)

    def test_unload(self):
        BROWN_CLUSTERS.unload()
        assert BROWN_CLUSTERS.resource is None
