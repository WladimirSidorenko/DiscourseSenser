#!/usr/bin/env python
# -*- coding: utf-8; mode: python; -*-

##################################################################
# Imports
from __future__ import absolute_import

from dsenser.constants import DEPS, PARSE_TREE, SENTENCES, WORDS
from dsenser.wang.implicit import WangImplicitSenser
import dsenser

from mock import patch
from pytest import fixture
from unittest import TestCase

##################################################################
# Constants
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
        "Connective": {},
        "Sense": [], "Type": "", "ID": 35709}

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
                 PARSE_TREE: PARSE_TREE_0_STR,
                 DEPS: [["det", "bill-2", "The-1"],
                        ["nsubj", "intends-3", "bill-2"],
                        ["root", "ROOT-0", "intends-3"],
                        ["aux", "restrict-5", "to-4"],
                        ["xcomp", "intends-3", "restrict-5"],
                        ["det", "RTC-7", "the-6"],
                        ["dobj", "restrict-5", "RTC-7"],
                        ["prep", "restrict-5", "to-8"],
                        ["nn", "borrowings-10", "Treasury-9"],
                        ["pobj", "to-8", "borrowings-10"],
                        ["advmod", "restrict-5", "only-11"],
                        ["mark", "receives-16", "unless-13"],
                        ["det", "agency-15", "the-14"],
                        ["nsubj", "receives-16", "agency-15"],
                        ["advcl", "restrict-5", "receives-16"],
                        ["amod", "authorization-19", "specific-17"],
                        ["amod", "authorization-19", "congressional-18"],
                        ["dobj", "receives-16", "authorization-19"]]}]}}


##################################################################
# Test Classes
class TestWangImplict(TestCase):

    @fixture(autouse=True)
    def set_ds(self):
        self.wis = WangImplicitSenser()

    def test_train(self):
        with patch("dsenser.wang.wangbase.WangBaseSenser.train",
                   autospec=True):
            self.wis.train([], {})

    def test_extract_features(self):
        feats = self.wis._extract_features(REL1, PARSE1)
        assert "Arg1First3-to_restrict_the" in feats and \
            feats["Arg1First3-to_restrict_the"] == 1
