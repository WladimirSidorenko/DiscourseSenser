#!/usr/bin/env python
# -*- coding: utf-8; mode: python; -*-

##################################################################
# Imports
from __future__ import absolute_import

from dsenser.nnbase import _norm_vec, _norm_word, NNBaseSenser
from dsenser.constants import CONNECTIVE, WORDS, SENTENCES, RAW_TEXT
import dsenser

from pytest import fixture
from mock import patch, MagicMock
from unittest import TestCase
import numpy as np
import pytest


##################################################################
# Constants
DOC_ID = "wsj_2200"
PARSE1 = {DOC_ID: {
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
                         [".", {"PartOfSpeech": "."}]]}]}}

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
        CONNECTIVE: {"CharacterSpanList": [[566, 572]], RAW_TEXT: "unless",
                     "TokenList": [[566, 572, 94, 2, 12]]},
        "Sense": [], "Type": "", "ID": 35709}


##################################################################
# Test Methods
def test_norm_vec():
    a = np.ones(10)
    b = _norm_vec(a)
    assert np.sqrt(np.sum(b**2)) == 1.


def test_norm_word():
    assert _norm_word("124345") == "1"
    assert _norm_word("ABCDEF") == "abcdef"


##################################################################
# Test Classes
class NNBase(NNBaseSenser):
    def _init_nn(self):
        """Initialize neural network.

        """
        super(NNBase, self)._init_nn()


class TestNNBaseSenser(TestCase):
    @fixture(autouse=True)
    def set_nnbase(self):
        self.nnbs = NNBase()

    def test_init(self):
        assert self.nnbs

    def test_init_lstsq(self):
        nnbs = NNBase(a_lstsq=True)
        with patch.object(dsenser.word2vec, "Word2Vec"):
            assert nnbs.w2v is not None

    def test_init_nn(self):
        with pytest.raises(NotImplementedError):
            self.nnbs._init_nn()

    def test_predict_0(self):
        with patch.multiple(self.nnbs,
                            _init_funcs=MagicMock(),
                            _init_wemb_funcs=MagicMock(),
                            _rel2x=MagicMock(return_value=None)):
            ret = [1]
            self.nnbs.predict(None, (None, None), ret, 0)
            assert ret[0] == 0

    def test_rel2x_0(self):
        with patch.object(self.nnbs,
                          "_arg2emb_idx", MagicMock(return_value=None)):
            assert self.nnbs._rel2x(REL1, PARSE1, lambda: None, lambda: None) \
                is None

    def test_rel2x_1(self):
        W2I = {}
        i = 0
        for w in PARSE1[DOC_ID][SENTENCES][2][WORDS]:
            if w[0] not in W2I:
                W2I[w[0].lower()] = i
                i += 1
        C2I = {"unless": 0}
        ret = self.nnbs._rel2x(
            REL1, PARSE1, lambda w: W2I[w], lambda w: C2I[w])
        assert np.allclose(ret[0], [3, 4, 0, 5, 3, 6, 7, 8])
        assert np.allclose(ret[1], [0, 11, 12, 13, 14, 15])
        assert ret[2] == 0

    def test_free(self):
        self.nnbs._free()
        assert self.nnbs.n_y < 0
        assert self.nnbs._w_stat is None
        assert len(self.nnbs._params) == 0

    def test_init_w_emb(self):
        with patch.multiple(self.nnbs, _params=[], ndim=3):
            self.nnbs._init_w_emb()
            assert len(self.nnbs._params) == 1
