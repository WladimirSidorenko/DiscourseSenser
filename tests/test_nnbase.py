#!/usr/bin/env python
# -*- coding: utf-8; mode: python; -*-

##################################################################
# Imports
from __future__ import absolute_import

from dsenser.nnbase import _norm_vec, _norm_word, NNBaseSenser, DFLT_VDIM
from dsenser.constants import CONNECTIVE, WORDS, SENTENCES, RAW_TEXT
import dsenser
import dsenser.nnbase

from pytest import fixture
from mock import patch, MagicMock, Mock
from unittest import TestCase
import numpy as np
import pytest
import theano


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

VEC0 = np.array([0, 0, 0, 0])
VEC1 = np.array([0, 1, 2, 3])
VEC2 = np.array([4, 5, 6, 7])
VEC3 = np.array([8, 9, 10, 11])


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
    EMB_ARG1 = Mock()
    W_INDICES_ARG1 = Mock()
    EMB_ARG2 = Mock()
    W_INDICES_ARG2 = Mock()
    CONN_INDEX = Mock()
    Y_gold = Mock()
    Y_pred = Mock()
    _predict_func_emb = None

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

    def test_predict_1(self):
        x = np.array([[0, 1, 2, 3, 4], [[0, 1, 2, 3, 4]]])
        y = np.array([0, 0, 0, 0])

        def _predict_func_mock(*args, **kwargs):
            return np.array([[0, 0, 1, 0],
                             [0, 0, 0, 1],
                             [0, 0, 1, 0],
                             [1, 0, 0, 0],
                             ])

        with patch.multiple(self.nnbs,
                            _predict_func=_predict_func_mock,
                            get_test_w_emb_i=None,
                            _init_wemb_funcs=MagicMock(),
                            _rel2x=MagicMock(return_value=x)):
            ret = np.array([[0] * 4])
            self.nnbs.predict(None, (None, None), ret, 0)
            assert np.allclose(ret[0], y)

    def test_cleanup(self):
        a = theano.shared(np.ones((10, 30)).astype(theano.config.floatX))
        assert np.allclose(a.get_value(), 1)
        self.nnbs._cleanup([[a]])
        assert np.allclose(a.get_value(), 0)

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

    def test_get_train_w_emb_i(self):
        with patch.multiple(self.nnbs,
                            w2emb_i={'1': 1, "hello": 2},
                            _w_stat={"world": 1, "z": 3},
                            w_i=3):
            with patch("dsenser.nnbase.UNK_PROB",
                       MagicMock(return_value=True)):
                assert self.nnbs._get_train_w_emb_i("1024") == 1
                assert self.nnbs._get_train_w_emb_i("HELLO") == 2
                assert self.nnbs._get_train_w_emb_i("world") == \
                    self.nnbs.unk_w_i
                assert self.nnbs._get_train_w_emb_i("Z") == 3
                assert self.nnbs.w_i == 4

    def test_get_test_w_emb_i(self):
        with patch.object(self.nnbs,
                          "w2emb_i", {'1': 1, "hello": 2}):
            assert self.nnbs._get_test_w_emb_i("1024") == 1
            assert self.nnbs._get_test_w_emb_i("HELLO") == 2
            assert self.nnbs._get_test_w_emb_i("ZZZ") == self.nnbs.unk_w_i

    def test_get_train_w2v_emb_i(self):
        with patch.multiple(self.nnbs,
                            w2emb_i={'1': 1, "hello": 2},
                            w2v={"world": None},
                            w_i=3):
            assert self.nnbs._get_train_w2v_emb_i("1024") == 1
            assert self.nnbs._get_train_w2v_emb_i("HELLO") == 2
            assert self.nnbs._get_train_w2v_emb_i("world") == 3
            assert self.nnbs._get_train_w2v_emb_i("Z") == self.nnbs.unk_w_i
            assert self.nnbs.w_i == 4

    def test_get_test_w2v_emb_i(self):
        with patch.multiple(self.nnbs,
                            w2emb_i={'1': 1, "hello": 2},
                            w2v={"world": VEC3},
                            W_EMB=np.vstack([VEC0, VEC1, VEC2])):
            assert np.allclose(self.nnbs._get_test_w2v_emb_i("Z"), VEC0)
            assert np.allclose(self.nnbs._get_test_w2v_emb_i("256"), VEC1)
            assert np.allclose(self.nnbs._get_test_w2v_emb_i("HELLO"), VEC2)
            assert np.allclose(self.nnbs._get_test_w2v_emb_i("world"), VEC3)

    def test_get_train_c_emb_i_0(self):
        i = 2
        conn2emb = {"when": i}
        with patch.object(self.nnbs,
                          "c2emb_i", conn2emb):
            ret = self.nnbs.get_train_c_emb_i("USUALLY WHEN")
            assert ret == i

    def test_get_train_c_emb_i_1(self):
        conn2emb = {}
        CONN = "USUALLY WHEN"
        with patch.multiple(self.nnbs,
                            c_i=2,
                            c2emb_i=conn2emb):
            ret = self.nnbs.get_train_c_emb_i(CONN)
            assert ret == 2
            assert self.nnbs.c_i == 3

    def test_get_test_c_emb_i_0(self):
        i = 3
        conn2emb = {"when": i}
        with patch.object(self.nnbs,
                          "c2emb_i", conn2emb):
            ret = self.nnbs.get_test_c_emb_i("USUALLY WHEN")
            assert ret == i

    def test_get_test_c_emb_i_1(self):
        i = 0
        conn2emb = {}
        with patch.multiple(self.nnbs,
                            c2emb_i=conn2emb,
                            unk_c_i=i,
                            ):
            ret = self.nnbs.get_test_c_emb_i("USUALLY")
            assert ret == i

    def test_init_w_emb(self):
        with patch.multiple(self.nnbs, _params=[], ndim=3):
            self.nnbs._init_w_emb()
            assert len(self.nnbs._params) == 1

    def test_init_wemb_funcs_0(self):
        word2vec = Mock()
        word2vec.load = MagicMock()
        word2vec.ndim = 2
        fmock = MagicMock()
        with patch.multiple(self.nnbs, _plain_w2v=True, ndim=8,
                            w2v=word2vec, _trained=True,
                            _predict_func_emb=fmock):
            self.nnbs._init_wemb_funcs()
            assert word2vec.load.called
            assert self.nnbs.ndim == word2vec.ndim
            assert self.nnbs.get_train_w_emb_i == \
                self.nnbs._get_train_w2v_emb_i
            assert self.nnbs.get_test_w_emb_i == \
                self.nnbs._get_test_w2v_emb_i
            assert self.nnbs._predict_func == fmock

    def test_init_wemb_funcs_1(self):
        word2vec = Mock()
        with patch.multiple(self.nnbs, _plain_w2v=True, w2v=None,
                            _trained=False):
            with patch.object(dsenser.nnbase, "Word2Vec", word2vec):
                self.nnbs._init_wemb_funcs()
                assert self.nnbs.get_test_w_emb_i == \
                    self.nnbs._get_train_w2v_emb_i

    def test_init_wemb_funcs_2(self):
        word2vec = Mock()
        word2vec.load = MagicMock()
        word2vec.ndim = 2
        fmock = MagicMock()
        with patch.multiple(self.nnbs, _plain_w2v=False, lstsq=True,
                            ndim=8, w2v=None, _trained=True,
                            _predict_func_emb=fmock):
            with patch.object(dsenser.nnbase, "Word2Vec", word2vec):
                self.nnbs._init_wemb_funcs()
                assert self.nnbs.ndim == DFLT_VDIM
                assert word2vec.load.called
                assert self.nnbs.get_train_w_emb_i == \
                    self.nnbs._get_train_w2v_emb_i
                assert self.nnbs.get_test_w_emb_i == \
                    self.nnbs._get_test_w2v_lstsq_emb_i

    def test_compute_w_stat(self):
        with patch.object(self.nnbs, "_w_stat"):
            self.nnbs._compute_w_stat(PARSE1)

    def test_init_funcs_0(self):
        rmsmock = MagicMock(return_value=(1, 2, 3))
        theano_mock = Mock()
        theano_mock.function = MagicMock()
        with patch.multiple(dsenser.nnbase, rmsprop=rmsmock,
                            theano=theano_mock):
                self.nnbs._init_funcs(True)
                assert rmsmock.called
