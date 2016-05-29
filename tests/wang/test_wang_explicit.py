#!/usr/bin/env python
# -*- coding: utf-8; mode: python; -*-

##################################################################
# Imports
from __future__ import absolute_import

from dsenser.constants import ARG1, CHAR_SPAN, CONNECTIVE, RAW_TEXT, \
    TOK_LIST, POS, WORDS
from dsenser.wang.explicit import DFLT_PRNT, LEFT, RIGHT, \
    WangExplicitSenser

from nltk import Tree
from pytest import fixture
from unittest import TestCase

import numpy as np


##################################################################
# Constants
TOKS = [[0, 1, 2, 3, 4], [0, 1, 2, 3, 5], [0, 1, 2, 3, 6]]
REL = {ARG1: {TOK_LIST: TOKS},
       CONNECTIVE: {CHAR_SPAN: [[566, 572]],
                    RAW_TEXT: "unless", TOK_LIST: [[566, 572, 94, 2, 12]]}}
TOK_0 = [(t, None) for t in "you do n't really have"
         " to".split(' ')]
MOD_0 = [0, 0, 0, 0, 0, 0, 1]
TOK_1 = [(t, None) for t in "you might or can do"
         " it".split(' ')]
MOD_1 = [1, 1, 0, 0, 0, 0, 0]
NEG_0_KEY = "Neg1-neg"
NEG_1_KEY = "Neg2-pos"
NEG_JNT_KEY = "JointNeg-neg|pos"

PARSES = [{}, {}, {}, {WORDS: [("One", {POS: "CC"}),
                               ("Two", {POS: "CC"}),
                               ("THREE", {POS: "CC"}),
                               ("Four", {POS: "CC"}),
                               ("Five", {POS: "CC"}),
                               ("SiX", {POS: "CC"}),
                               ("SEVEN", {POS: "CC"})]}]
PARSE_TREE_0 = Tree.fromstring("( (S (NP (DT The) (NN bill)) "
                               "(VP (VBZ intends) (S (VP (TO to)"
                               " (VP (VB restrict) (NP (DT the)"
                               " (NNP RTC)) (PP (TO to) (NP (NNP"
                               " Treasury) (NNS borrowings))) (ADVP"
                               " (RB only)) (, ,) (SBAR (IN unless)"
                               " (S (NP (DT the) (NN agency)) (VP (VBZ"
                               " receives) (NP (JJ specific) (JJ"
                               " congressional) (NN authorization)))))))))"
                               " (. .)) )")
TOKS_0 = {"words": [["The", {"PartOfSpeech": "DT"}],
                    ["bill", {"PartOfSpeech": "NN"}],
                    ["intends", {"PartOfSpeech": "VBZ"}],
                    ["to", {"PartOfSpeech": "TO"}],
                    ["restrict", {"PartOfSpeech": "VB"}],
                    ["the", {"PartOfSpeech": "DT"}],
                    ["RTC", {"PartOfSpeech": "NNP"}],
                    ["to", {"PartOfSpeech": "TO"}],
                    ["Treasury", {"PartOfSpeech": "NNP"}],
                    ["borrowings", {"PartOfSpeech": "NNS"}],
                    ["only", {"PartOfSpeech": "RB"}],
                    [",", {"PartOfSpeech": ","}],
                    ["unless", {"PartOfSpeech": "IN"}],
                    ["the", {"PartOfSpeech": "DT"}],
                    ["agency", {"PartOfSpeech": "NN"}],
                    ["receives", {"PartOfSpeech": "VBZ"}],
                    ["specific", {"PartOfSpeech": "JJ"}],
                    ["congressional", {"PartOfSpeech": "JJ"}],
                    ["authorization", {"PartOfSpeech": "NN"}],
                    [".", {"PartOfSpeech": "."}]]}

PARSE_TREE_1 = Tree.fromstring("(S (DT The))")
PARSE_TREE_2 = Tree.fromstring("(S The)")


##################################################################
# Test Classes
class TestWangExplict(TestCase):
    @fixture(autouse=True)
    def set_ds(self):
        self.wes = WangExplicitSenser()

    def test_get_sib_left_0(self):
        t_ids = [0]
        sib = self.wes._get_sib(t_ids, PARSE_TREE_0, LEFT)
        assert sib == "NONE"

    def test_get_sib_left_1(self):
        t_ids = [4]
        sib = self.wes._get_sib(t_ids, PARSE_TREE_0, LEFT)
        assert sib == "TO"

    def test_get_sib_left_2(self):
        t_ids = [2]
        sib = self.wes._get_sib(t_ids, PARSE_TREE_0, LEFT)
        assert sib == "NP"

    def test_get_sib_right_0(self):
        t_ids = [t[-1] for t in REL[CONNECTIVE][TOK_LIST]]
        sib = self.wes._get_sib(t_ids, PARSE_TREE_0, RIGHT)
        assert sib == "NONE"

    def test_get_sib_right_1(self):
        t_ids = [0]
        sib = self.wes._get_sib(t_ids, PARSE_TREE_0, RIGHT)
        assert sib == "VP"

    def test_get_prev_conn_0(self):
        conn_t_ids = [t[-1] for t in REL[CONNECTIVE][TOK_LIST]]
        sib = self.wes._get_prev_conn(conn_t_ids[0], PARSE_TREE_0, TOKS_0)
        assert sib == ([], [])

    def test_get_ctx_nodes_0(self):
        t_ids = [0]
        ctx_nodes = self.wes._get_ctx_nodes(t_ids, PARSE_TREE_2)
        assert ctx_nodes == [DFLT_PRNT]

    def test_get_ctx_nodes_1(self):
        conn_t_ids = [t[-1] for t in REL[CONNECTIVE][TOK_LIST]]
        ctx_nodes = self.wes._get_ctx_nodes(conn_t_ids, PARSE_TREE_0)
        assert ctx_nodes == ['VP', 'VB', 'NP', 'PP', 'ADVP', ',', 'SBAR']

    def test_get_path_0(self):
        conn_t_ids = [t[-1] for t in REL[CONNECTIVE][TOK_LIST]]
        inode_id = self.wes._get_path(conn_t_ids, PARSE_TREE_0)
        assert inode_id == (0, 1, 1, 0, 1, 5)

    def test_get_path_1(self):
        t_ids = [0, 1]
        inode_id = self.wes._get_path(t_ids, PARSE_TREE_0)
        assert inode_id == (0, 0)

    def test_get_path_2(self):
        t_ids = [19]
        inode_id = self.wes._get_path(t_ids, PARSE_TREE_0)
        assert inode_id == (0,)

    def test_get_path_3(self):
        t_ids = [0]
        inode_id = self.wes._get_path(t_ids, PARSE_TREE_1)
        assert inode_id == (0,)

    def test_get_negation_0(self):
        ret = {}
        self.wes._get_negation(ret, TOK_0, TOK_1)
        assert NEG_0_KEY in ret and ret[NEG_0_KEY] == 1
        assert NEG_1_KEY in ret and ret[NEG_1_KEY] == 1
        assert NEG_JNT_KEY in ret and ret[NEG_JNT_KEY] == 1

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
