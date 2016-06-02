#!/usr/bin/env python
# -*- coding: utf-8; mode: python; -*-

##################################################################
# Imports
from __future__ import absolute_import

from dsenser.constants import ARG1, CHAR_SPAN, CONNECTIVE, PARSE_TREE, \
    RAW_TEXT, TOK_LIST, POS, SENTENCES, WORDS
from dsenser.wang.explicit import DFLT_PRNT, LEFT, RIGHT, PREV_NONE, \
    WHEN, WangExplicitSenser
import dsenser

from copy import deepcopy
from mock import patch
from nltk import Tree
from pytest import fixture
from unittest import TestCase

import numpy as np


##################################################################
# Constants
PARSES = [{}, {}, {}, {WORDS: [("One", {POS: "CC"}),
                               ("Two", {POS: "CC"}),
                               ("THREE", {POS: "CC"}),
                               ("Four", {POS: "CC"}),
                               ("Five", {POS: "CC"}),
                               ("SiX", {POS: "CC"}),
                               ("SEVEN", {POS: "CC"})]}]

PARSE_TREE_0_STR = "( (S (NP (DT The) (NN bill)) " \
                   "(VP (VBZ intends) (S (VP (TO to)" \
                   " (VP (VB restrict) (NP (DT the)" \
                   " (NNP RTC)) (PP (TO to) (NP (NNP" \
                   " Treasury) (NNS borrowings))) (ADVP" \
                   " (RB only)) (, ,) (SBAR (IN unless)" \
                   " (S (NP (DT the) (NN agency)) (VP (VBZ" \
                   " receives) (NP (JJ specific) (JJ" \
                   " congressional) (NN authorization)))))))))" \
                   " (. .)) )"
PARSE_TREE_0 = Tree.fromstring(PARSE_TREE_0_STR)
PARSE_TREE_1 = Tree.fromstring("(S (DT The))")
PARSE_TREE_2 = Tree.fromstring("(S The)")

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

TOK_0 = [(t, None) for t in "you do n't really have"
         " to".split(' ')]
MOD_0 = [0, 0, 0, 0, 0, 0, 1]
TOK_1 = [(t, None) for t in "you might or can do"
         " it".split(' ')]
MOD_1 = [1, 1, 0, 0, 0, 0, 0]
NEG_0_KEY = "Neg1-neg"
NEG_1_KEY = "Neg2-pos"
NEG_JNT_KEY = "JointNeg-neg|pos"

TOKS = [[0, 1, 2, 3, 4], [0, 1, 2, 3, 5], [0, 1, 2, 3, 6]]
REL = {ARG1: {TOK_LIST: TOKS},
       CONNECTIVE: {CHAR_SPAN: [[566, 572]],
                    RAW_TEXT: "unless", TOK_LIST: [[566, 572, 94, 2, 12]]}}
REL1 = {"DocID": "wsj_2200",
        "Arg1": {"CharacterSpanList": [[517, 564]],
                 "RawText": "to restrict the RTC to Treasury"
                 " borrowings only",
                 "TokenList": [[517, 519, 85, 2, 3], [520, 528, 86, 2, 4],
                               [529, 532, 87, 2, 5], [533, 536, 88, 2, 6],
                               [537, 539, 89, 2, 7], [540, 548, 90, 2, 8],
                               [549, 559, 91, 2, 9], [560, 564, 92, 2, 10]]},
        "Arg2": {"CharacterSpanList": [[573, 629]], "RawText": "the agency"
                 " receives specific congressional authorization",
                 "TokenList": [[573, 576, 95, 2, 13], [577, 583, 96, 2, 14],
                               [584, 592, 97, 2, 15], [593, 601, 98, 2, 16],
                               [602, 615, 99, 2, 17], [616, 629, 100, 2, 18]]},
        "Connective": {"CharacterSpanList": [[566, 572]], "RawText": "unless",
                       "TokenList": [[566, 572, 94, 2, 12]]},
        "Sense": [], "Type": "", "ID": 35709}

REL1_1 = deepcopy(REL1)
REL1_1["Connective"]["TokenList"][0][-2:] = [3, 0]

REL1_2 = deepcopy(REL1)
REL1_2["Connective"]["TokenList"][0][-2:] = [0, 0]

REL1_3 = deepcopy(REL1)
REL1_3["Connective"]["TokenList"][0][-2:] = [1, 0]

REL1_4 = deepcopy(REL1)
REL1_4["Connective"]["TokenList"][0][-2:] = [2, 20]

PARSE1 = {"wsj_2200": {
    SENTENCES: [{WORDS: []}, {WORDS: []},
                {WORDS: [["The", {"PartOfSpeech": "DT"}],
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
                         [".", {"PartOfSpeech": "."}]],
                 PARSE_TREE: PARSE_TREE_0_STR}]}}

PARSE2 = deepcopy(PARSE1)
PARSE2["wsj_2200"][SENTENCES][-1][PARSE_TREE] = "()"

PARSE3 = deepcopy(PARSE1)
PARSE3["wsj_2200"][SENTENCES][-1][WORDS][0][0] = "Unless"
PARSE3["wsj_2200"][SENTENCES][-1][WORDS][12][0] = WHEN
PARSE3["wsj_2200"][SENTENCES][-1][PARSE_TREE] = \
    "( (S (NP (DT Unless) (NN bill)) " \
    "(VP (VBZ intends) (S (VP (TO to)" \
    " (VP (VB restrict) (NP (DT the)" \
    " (NNP RTC)) (PP (TO to) (NP (NNP" \
    " Treasury) (NNS borrowings))) (ADVP" \
    " (RB only)) (, ,) (SBAR (IN when)" \
    " (S (NP (DT the) (NN agency)) (VP (VBZ" \
    " receives) (NP (JJ specific) (JJ" \
    " congressional) (NN authorization)))))))))" \
    " (. .)) )"
REL_3 = {"Arg1": {"CharacterSpanList": [[7392, 7429]],
                  "RawText": "But then it suddenly burst upward 7.5",
                  "TokenList": [[7392, 7395, 1455, 70, 0],
                                [7396, 7400, 1456, 70, 1],
                                [7401, 7403, 1457, 70, 2],
                                [7404, 7412, 1458, 70, 3],
                                [7413, 7418, 1459, 70, 4],
                                [7419, 7425, 1460, 70, 5],
                                [7426, 7429, 1461, 70, 6]]},
         "Arg2": {"CharacterSpanList": [[7433, 7500]],
                  "RawText":
                  "Goldman, Sachs & Co. stepped in and bought"
                  " almost every share offer",
                  "TokenList": [[7433, 7440, 1463, 70, 8],
                                [7440, 7441, 1464, 70, 9],
                                [7442, 7447, 1465, 70, 10],
                                [7448, 7449, 1466, 70, 11],
                                [7450, 7453, 1467, 70, 12],
                                [7454, 7461, 1468, 70, 13],
                                [7462, 7464, 1469, 70, 14],
                                [7465, 7468, 1470, 70, 15],
                                [7469, 7475, 1471, 70, 16],
                                [7476, 7482, 1472, 70, 17],
                                [7483, 7488, 1473, 70, 18],
                                [7489, 7494, 1474, 70, 19],
                                [7495, 7500, 1475, 70, 20]]},
         "Connective": {"CharacterSpanList": [[7430, 7432]],
                        "RawText": "as",
                        "TokenList": [[7430, 7432, 1462, 70, 7]]},
         "DocID": "wsj_2276", "ID": 37022,
         "Sense": ["Contingency.Cause.Reason"], "Type": "Explicit"}
PARSE_TREE_3 = Tree.fromstring(
    "( (S (S (CC But) (ADVP (RB then)) (NP (PRP it))"
    " (VP (ADVP (RB suddenly)) (VP (VBP burst) (ADVP"
    " (RB upward))) (NP (CD 7.5)) (SBAR (IN as) "
    "(S (NP (NNP Goldman) (, ,) (NNP Sachs) (CC &)"
    " (NNP Co.)) (VP (VP (VBD stepped) (PP (IN in)))"
    " (CC and) (VP (VBD bought) (NP (RB almost)"
    " (DT every) (NN share) (NN offer)))))))) (, ,)"
    " (NP (NNS traders)) (VP (VBD said)) (. .)) )"
    )
TOKS_3 = [["But", {"PartOfSpeech": "CC"}],
          ["then", {"PartOfSpeech": "RB"}],
          ["it", {"PartOfSpeech": "PRP"}],
          ["suddenly", {"PartOfSpeech": "RB"}],
          ["burst", {"PartOfSpeech": "VBP"}],
          ["upward", {"PartOfSpeech": "RB"}],
          ["7.5", {"PartOfSpeech": "CD"}],
          ["as", {"PartOfSpeech": "IN"}],
          ["Goldman", {"PartOfSpeech": "NNP"}],
          [",", {"PartOfSpeech": ","}],
          ["Sachs", {"PartOfSpeech": "NNP"}],
          ["&", {"PartOfSpeech": "CC"}],
          ["Co.", {"PartOfSpeech": "NNP"}],
          ["stepped", {"PartOfSpeech": "VBD"}],
          ["in", {"PartOfSpeech": "IN"}],
          ["and", {"PartOfSpeech": "CC"}],
          ["bought", {"PartOfSpeech": "VBD"}],
          ["almost", {"PartOfSpeech": "RB"}],
          ["every", {"PartOfSpeech": "DT"}],
          ["share", {"PartOfSpeech": "NN"}],
          ["offer", {"PartOfSpeech": "NN"}],
          [",", {"PartOfSpeech": ","}],
          ["traders", {"PartOfSpeech": "NNS"}],
          ["said", {"PartOfSpeech": "VBD"}],
          [".", {"PartOfSpeech": "."}]]

PARSE4 = Tree.fromstring("( (S (NP (DT If) (NN bill)) "
                         "(VP (VBZ intends) (S (VP (IN then)"
                         " (VP (VB restrict) (NP (DT the)"
                         " (NNP RTC)) (PP (TO to) (NP (NNP"
                         " Treasury) (NNS borrowings))) (ADVP"
                         " (RB only)) (, ,) (SBAR (IN unless)"
                         " (S (NP (DT the) (NN agency)) (VP (VBZ"
                         " receives) (NP (JJ specific) (JJ"
                         " congressional) (NN authorization)))))))))"
                         " (. .)) )"
                         )
TOKS_4 = deepcopy(TOKS_3)
TOKS_4[0] = ["If", {"PartOfSpeech": "DT"}]
TOKS_4[3] = ["then", {"PartOfSpeech": "IN"}]

PARSE5 = Tree.fromstring("( (S (NP (DT If) (NN bill)) "
                         "(VP (VBZ intends) (S (VP (TO to)"
                         " (VP (VB restrict) (NP (DT the)"
                         " (NNP RTC)) (PP (TO to) (NP (NNP"
                         " Treasury) (NNS borrowings))) (ADVP"
                         " (RB only)) (, ,) (SBAR (IN unless)"
                         " (S (NP (DT the) (NN agency)) (VP (VBZ"
                         " receives) (NP (JJ specific) (JJ"
                         " congressional) (NN authorization)))))))))"
                         " (. .)) )"
                         )
TOKS_5 = deepcopy(TOKS_3)
TOKS_5[0] = ["If", {"PartOfSpeech": "DT"}]

PARSE6 = Tree.fromstring("( (S (NP (IN Then) (NN bill)) "
                         "(VP (IN if) (S (VP (IN then)"
                         " (VP (VB restrict) (NP (DT the)"
                         " (NNP RTC)) (PP (TO to) (NP (NNP"
                         " Treasury) (NNS borrowings))) (ADVP"
                         " (RB only)) (, ,) (SBAR (IN unless)"
                         " (S (NP (DT the) (NN agency)) (VP (VBZ"
                         " receives) (NP (JJ specific) (JJ"
                         " congressional) (NN authorization)))))))))"
                         " (. .)) )"
                         )
TOKS_6 = deepcopy(TOKS_3)
TOKS_6[0] = ["Then", {"PartOfSpeech": "IN"}]
TOKS_6[2] = ["if", {"PartOfSpeech": "IN"}]
TOKS_6[3] = ["then", {"PartOfSpeech": "IN"}]

REL7 = deepcopy(REL)
REL7[CONNECTIVE][TOK_LIST][0][-1] = 19
PARSE7 = Tree.fromstring("( (S (NP (IN Unless) (NN bill)) "
                         "(VP (VBZ intends) (S (VP (TO to)"
                         " (VP (VB restrict) (NP (DT the)"
                         " (NNP RTC)) (PP (TO to) (NP (NNP"
                         " Treasury) (NNS borrowings))) (ADVP"
                         " (RB only)) (, ,) (SBAR (IN as)"
                         " (S (NP (DT the) (NN agency)) (VP (VBZ"
                         " receives) (NP (JJ specific) (JJ"
                         " congressional) (IN if)))))))))"
                         " (IN unless)) )"
                         )
TOKS_7 = deepcopy(PARSE1["wsj_2200"][SENTENCES][2][WORDS])
TOKS_7[0][1]["PartOfSpeech"] = "IN"
TOKS_7[-2] = ["if", {"PartOfSpeech": "IN"}]
TOKS_7[-1] = ["unless", {"PartOfSpeech": "IN"}]


##################################################################
# Test Classes
class TestWangExplict(TestCase):
    @fixture(autouse=True)
    def set_ds(self):
        self.wes = WangExplicitSenser()

    def test_train(self):
        with patch("dsenser.wang.wangbase.WangBaseSenser.train",
                   autospec=True):
            self.wes.train([], {})

    def test_train(self):
        with patch("dsenser.wang.wangbase.WangBaseSenser.train",
                   autospec=True):
            self.wes.train([], {})

    def test_extract_features_0(self):
        feats = self.wes._extract_features(REL1, PARSE1)
        assert "ConStr-unless" in feats and feats["ConStr-unless"] == 1

    def test_extract_features_1(self):
        feats = self.wes._extract_features(REL1, PARSE2)
        assert "ConStr-unless" in feats and feats["ConStr-unless"] == 1

    def test_extract_features_2(self):
        feats = self.wes._extract_features(REL1, PARSE3)
        assert "when-unless" in feats
        assert "when-" not in feats

    def test_get_conn_pos(self):
        conn_txt = self.wes._get_conn_pos("wsj_2200",
                                          REL1, PARSE1)
        assert conn_txt == "IN"

    def test_get_conn_prev_0(self):
        assert self.wes._get_conn_prev("wsj_2200", REL1, PARSE1) == \
            ','

    def test_get_conn_prev_1(self):
        assert self.wes._get_conn_prev("wsj_2200", REL1_1, PARSE1) == \
            '.'

    def test_get_conn_prev_2(self):
        assert self.wes._get_conn_prev("wsj_2200", REL1_2, PARSE1) == \
            PREV_NONE

    def test_get_conn_succ_0(self):
        assert self.wes._get_conn_succ("wsj_2200", REL1, PARSE1) == \
            ("the", "DT")

    def test_get_conn_succ_1(self):
        assert self.wes._get_conn_succ("wsj_2200", REL1_3, PARSE1) == \
            ("The", "DT")

    def test_get_conn_succ_2(self):
        assert self.wes._get_conn_succ("wsj_2200", REL1_4, PARSE1) == \
            ("succ_none", "succ_tok_none")

    def test_get_cat(self):
        t_ids = [t[-1] for t in REL[CONNECTIVE][TOK_LIST]]
        cat = self.wes._get_cat(t_ids, PARSE_TREE_0)
        assert cat == "SBAR"

    def test_get_prnt_cat_0(self):
        t_ids = [t[-1] for t in REL[CONNECTIVE][TOK_LIST]]
        cat = self.wes._get_prnt_cat(t_ids, PARSE_TREE_0)
        assert cat == "VP"

    def test_get_prnt_cat_1(self):
        t_ids = [0, 1]
        cat = self.wes._get_prnt_cat(t_ids, PARSE_TREE_0)
        assert cat == DFLT_PRNT

    def test_get_prnt_cat_2(self):
        t_ids = [0]
        cat = self.wes._get_prnt_cat(t_ids, PARSE_TREE_2)
        assert cat == DFLT_PRNT

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
        prev_conns = self.wes._get_prev_conn(conn_t_ids[0],
                                             PARSE_TREE_0, TOKS_0)
        assert prev_conns == ([], [])

    def test_get_prev_conn_1(self):
        conn_t_ids = [t[-1] for t in REL_3[CONNECTIVE][TOK_LIST]]
        prev_conns = self.wes._get_prev_conn(conn_t_ids[0],
                                             PARSE_TREE_3, TOKS_3)
        assert prev_conns == ([(("but",),), (("then",),)],
                              [["CC"], ["RB"]])

    def test_get_prev_conn_2(self):
        conn_t_ids = [t[-1] for t in REL[CONNECTIVE][TOK_LIST]]
        prev_conns = self.wes._get_prev_conn(conn_t_ids[0],
                                             PARSE4, TOKS_4)
        assert prev_conns == \
            ([(("if",), ("then",)), (("if",),), (("then",),)],
             [["DT", "IN"], ["DT"], ["IN"]])

    def test_get_prev_conn_3(self):
        conn_t_ids = [t[-1] for t in REL[CONNECTIVE][TOK_LIST]]
        prev_conns = self.wes._get_prev_conn(conn_t_ids[0],
                                             PARSE5, TOKS_5)
        assert prev_conns == ([(("if",),)], [["DT"]])

    def test_get_prev_conn_4(self):
        conn_t_ids = [t[-1] for t in REL[CONNECTIVE][TOK_LIST]]
        prev_conns = self.wes._get_prev_conn(conn_t_ids[0],
                                             PARSE6, TOKS_6)
        assert prev_conns == \
            ([((u'then',),), ((u'if',), (u'then',)),
              ((u'if',),), ((u'then',),)],
             [['IN'], ['IN', 'IN'], ['IN'], ['IN']])

    def test_get_prev_conn_5(self):
        conn_t_ids = [t[-1] for t in REL7[CONNECTIVE][TOK_LIST]]
        prev_conns = self.wes._get_prev_conn(conn_t_ids[0],
                                             PARSE7, TOKS_7)
        assert prev_conns == ([(('unless',),), (('as',),), (('if',),)],
                              [['IN'], ['IN'], ['IN']])

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
