#!/usr/bin/env python
# -*- coding: utf-8; mode: python; -*-

##################################################################
# Imports
from __future__ import absolute_import

from dsenser.constants import ARG1, TOK_LIST, POS, WORDS
from dsenser.wang.explicit import WangExplicitSenser

from pytest import fixture
from unittest import TestCase
from mock import patch

import numpy as np


##################################################################
# Constants
TOKS = [[0, 1, 2, 3, 4], [0, 1, 2, 3, 5], [0, 1, 2, 3, 6]]
REL = {ARG1: {TOK_LIST: TOKS}}
PARSES = [{}, {}, {}, {WORDS: [("One", {POS: "CC"}),
                               ("Two", {POS: "CC"}),
                               ("THREE", {POS: "CC"}),
                               ("Four", {POS: "CC"}),
                               ("Five", {POS: "CC"}),
                               ("SiX", {POS: "CC"}),
                               ("SEVEN", {POS: "CC"})]}]
TOK_0 = [(t, None) for t in "you do n't really have"
         " to".split(' ')]
MOD_0 = [0, 0, 0, 0, 0, 0, 1]
TOK_1 = [(t, None) for t in "you might or can do"
         " it".split(' ')]
MOD_1 = [1, 1, 0, 0, 0, 0, 0]


##################################################################
# Test Classes
class TestWangExplict(TestCase):
    @fixture(autouse=True)
    def set_ds(self):
        self.wes = WangExplicitSenser()

    def test_get_arg_negation_0(self):
        assert self.wes._get_arg_negation(TOK_0) == "neg"

    def test_get_arg_negation_1(self):
        assert self.wes._get_arg_negation(TOK_1) == "pos"

    def test_get_modality(self):
        ret = {}
        self.wes._get_modality(ret, TOK_0, TOK_1)
        mod1_ft = "Mod1-" + ''.join(str(i) for i in MOD_0)
        mod2_ft = "Mod2-" + ''.join(str(i) for i in MOD_1)
        assert mod1_ft in ret and ret[mod1_ft] == 1
        assert mod2_ft in ret and ret[mod2_ft] == 1

    def test_get_arg_modality_0(self):
        mod = self.wes._get_arg_modality(TOK_0)
        assert np.allclose(mod, MOD_0)

    def test_get_arg_modality_1(self):
        mod = self.wes._get_arg_modality(TOK_1)
        assert np.allclose(mod, MOD_1)

    def test_get_toks_pos(self):
        toks_pos = self.wes._get_toks_pos(PARSES, REL, ARG1)
        assert toks_pos == [("five", "CC"), ("six", "CC"), ("seven", "CC")]

    def test_get_snt2tok_0(self):
        snt2tok = self.wes._get_snt2tok([])
        assert len(snt2tok) == 0

    def test_get_snt2tok_1(self):
        snt2tok = self.wes._get_snt2tok(TOKS)
        assert snt2tok[3] == set([4, 5, 6])

    def test_escape_feat_0(self):
        assert self.wes._escape_feat("a 	b g ") == "a b g"

    def test_escape_feat_1(self):
        assert self.wes._escape_feat("a=1") == "a_1"
