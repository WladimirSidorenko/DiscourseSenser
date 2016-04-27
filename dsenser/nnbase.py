#!/usr/bin/env python
# -*- coding: utf-8; mode: python; -*-

"""Module providing abstract class for NN-based sense classification.

Attributes:
NNBaseSenser (class):
  abstract class defining interface for neural-network based sense
  classification

"""

##################################################################
# Imports
from __future__ import absolute_import, print_function

from dsenser.base import BaseSenser
from dsenser.constants import ARG1, ARG2, CONNECTIVE, DOC_ID, \
    RAW_TEXT, SENTENCES, SENSE, TOK_IDX, WORDS
from dsenser.resources import CHM
from dsenser.theano_utils import config, floatX, rmsprop, theano, \
    CONV_EPS, HE_UNIFORM, MAX_ITERS, TT, TRNG, DFLT_VDIM
from dsenser.word2vec import Word2Vec

from collections import Counter
from datetime import datetime
import abc
import numpy as np
import re
import sys


##################################################################
# Variables and Constants
np.random.seed()

MAX = 1e10
INF = float('inf')
UNK_PROB = lambda: np.random.binomial(1, 0.05)
DIG_RE = re.compile(r"^[\d.]*\d[\d.]*$")


##################################################################
# Methods
def _norm_vec(a_x):
    """Normali length of the inout vector.

    Args:
    a_x (np.array):
    input vector

    Returns:
    (np.array):
    normalized vector

    """
    return a_x / (np.linalg.norm(a_x) or MAX)


def _norm_word(a_word):
    """Normalize word.

    Args:
    a_word (str):
    word to normalize

    Returns:
    (str):
    normalized word form

    """
    return DIG_RE.sub('1', a_word.lower())


##################################################################
# Class
class NNBaseSenser(BaseSenser):
    """Abstract class for NN-based sense disambiguation.

    Attrs:
    n_y (int): number of distinct classes

    Methods:

    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, a_w2v=False, a_lstsq=False):
        """Class constructor.

        Args:
        a_w2v (bool):
          use pre-trained word2vec instance
        a_lstsq (bool):
          pre-train task-specific word embeddings, but use least-square method
          to generate embeddings for unknown words from generic word2vec
          vectors

        """
        # access to the original word2vec resource
        if a_lstsq:
            a_w2v = True
        if a_w2v:
            self.w2v = Word2Vec  # singleton object
        else:
            self.w2v = None
        self.lstsq = a_lstsq
        self._plain_w2v = self.w2v and not self.lstsq
        # matrix mapping word2vec to task-specific embeddings
        self.w2emb = None
        self.ndim = -1    # vector dimensionality will be initialized later
        self.intm_dim = -1
        # mapping from word to its embedding index
        self.unk_w_i = 0
        self._aux_keys = set((0, ))
        self.w_i = 1
        self.w2emb_i = dict()
        # mapping from connective to its embedding index
        self.unk_c_i = 0
        self.c_i = 1
        self.c2emb_i = dict()
        # variables needed for training
        self._trained = False
        self._params = []
        self._w_stat = None
        self.W_EMB = self.CONN_EMB = self._cost = None
        self.use_dropout = theano.shared(floatX(0.))
        # initialize theano functions to None
        self._reset_funcs()
        # set up functions for obtaining word embeddings at train and test
        # times
        self._init_wemb_funcs()

    def train(self, a_train_data, a_dev_data=None, a_n_y=-1,
              a_i=-1, a_train_out=None, a_dev_out=None):
        """Method for training the model.

        Args:
        a_train_data (2-tuple(list, dict)):
          list of training JSON data
        a_dev_data (2-tuple(list, dict) or None):
          list of development JSON data
        a_n_y (int):
          number of distinct classes
        a_i (int):
          row index for the output predictions
        a_train_out (np.array or None):
          predictions for the training set
        a_dev_out (np.array or None):
          predictions for the training set

        Returns:
        (void):
          updates ``a_train_out`` and ``a_dev_out`` in place

        """
        self.n_y = a_n_y
        # allocate data to development set if there is none
        if a_dev_data is None or not a_dev_data[0]:
            train_rels, parses = a_train_data
            docs = parses.keys()
            n_docs = len(docs)
            n_dev = max(n_docs / 10, 1)
            # sample without replacement
            dev_docs = set(np.random.choice(docs, n_dev, False))
            new_train_rels, dev_rels = [], []
            for irel in train_rels:
                # relations are numbered at this place
                if irel[-1][DOC_ID] in dev_docs:
                    dev_rels.append(irel)
                else:
                    new_train_rels.append(irel)
            a_train_data = (new_train_rels, parses)
            a_dev_data = (dev_rels, parses)
        # convert training and development sets to features
        x_train, y_train = self._generate_ts(a_train_data,
                                             self.get_train_w_emb_i,
                                             self.get_train_c_emb_i)
        x_dev, y_dev = self._generate_ts(a_dev_data,
                                         self.get_test_w_emb_i,
                                         self.get_test_c_emb_i)
        # initialize the network
        self._init_nn()
        # activate dropout for training
        self.use_dropout.set_value(1.)

        # perform the training
        min_dev_cost = INF
        prev_train_cost = train_cost = dev_cost = 0.
        best_params = []
        for i in xrange(MAX_ITERS):
            train_cost = 0.
            start_time = datetime.utcnow()
            # perform one training iteration
            for (_, (emb1, emb2, conn)), y in zip(x_train, y_train):
                # print("emb1 =", repr(emb1), file=sys.stderr)
                # print("emb2 =", repr(emb2), file=sys.stderr)
                # print("self._debug_nn =", repr(self._debug_nn(emb1, emb2)),
                #       file=sys.stderr)
                # sys.exit(66)
                train_cost += self._grad_shared(emb1, emb2, conn, y)
                self._update()
            # estimate the model on the dev set
            dev_cost = 0.
            # temporarily deactivate dropout
            self.use_dropout.set_value(0.)
            for (_, (emb1, emb2, conn)), y in zip(x_dev, y_dev):
                dev_cost += self._compute_cost(emb1, emb2, conn, y)
            # switch dropout on again
            self.use_dropout.set_value(1.)
            end_time = datetime.utcnow()
            time_delta = (end_time - start_time).seconds
            if min_dev_cost == INF or dev_cost < min_dev_cost:
                best_params = [p.get_value() for p in self._params]
                min_dev_cost = dev_cost
            print("Iteration {:d}:\ttrain_cost = {:f}\t"
                  "dev_cost={:f}\t({:.2f} sec)".format(
                      i, train_cost, dev_cost, time_delta), file=sys.stderr)
            if abs(prev_train_cost - train_cost) < CONV_EPS:
                break
            prev_train_cost = train_cost
        # deactivate dropout
        self.use_dropout.set_value(0.)
        if best_params:
            for p, val in zip(self._params, best_params):
                p.set_value(val)
        # make predictions for the judge
        if a_i >= 0:
            # deactivate dropout once again
            if a_train_out is not None:
                for i, x_i in x_train:
                    self._predict(x_i, a_train_out[i], a_i)
            if a_dev_out:
                for i, x_i in x_dev:
                    self._predict(x_i, a_dev_out[i], a_i)
            else:
                for i, x_i in x_dev:
                    self._predict(x_i, a_train_out[i], a_i)
        # reset function members to allow cPickle store this model
        self._reset_funcs()
        self._cleanup(self._rms_params)
        if self.w2v:
            self.W_EMB = floatX(self.W_EMB.get_value())
        if self.lstsq:
            self.w2v.load()
            self._init_w2emb()
        if self.w2v is not None:
            self.w2v.unload()
            self.w2v = None
        self._trained = True

    @abc.abstractmethod
    def _init_nn(self):
        """Initialize neural network.

        Args:
        (void)

        Returns:
        (void)

        """
        raise NotImplementedError

    def predict(self, a_rel, a_data, a_ret, a_i):
        """Method for predicting sense of single relation.

        Args:
        a_rel (dict):
          discourse relation whose sense should be predicted
        a_data (2-tuple(dict, dict)):
          list of input JSON data
        a_ret (np.array):
          output prediction vector
        a_i (int):
          row index in the output vector

        Returns:
        (void):
          updates ``a_ret`` in place

        """
        parses = a_data[-1]
        # do not change this order as `_init_wemb_funcs()` might override
        # `_init_funcs()`
        if self._predict_func is None:
            self._init_funcs()
        if self.get_test_w_emb_i is None:
            self._init_wemb_funcs()
        input_args = self._rel2x(a_rel, parses,
                                 self.get_test_w_emb_i,
                                 self.get_test_c_emb_i)
        return self._predict(input_args, a_ret, a_i)

    def _predict(self, a_args, a_ret, a_i):
        """Method for predicting sense of single relation.

        Args:
        a_args (list):
          list of input arguments to the prediction function
        a_ret (np.array):
          output prediction vector
        a_i (int):
          row index in the output vector

        Returns:
        (void):
          updates ``a_ret`` in place

        """
        # obtain model's estimates
        dec = self._predict_func(*a_args)
        if len(dec.shape) > 1:
            dec = np.mean(dec, axis=0)
        for i, ival in enumerate(dec):
            a_ret[a_i][i] = ival

    def _free(self):
        """Free resources used by the model.

        Args:
        (void):

        Returns:
        (void):

        """
        self.n_y = -1
        self._w_stat = None
        self._cleanup(self._params)
        self._params = []

    def _cleanup(self, a_vars):
        """Clean-up memory occupied by shared variables.

        Args:
        a_shared_vars: list(theano.shared)
          list of shared variables whose memory should be freed

        Returns:
        (void)

        """
        dim = 0
        for var_list in a_vars:
            for v in var_list:
                dim = len(v.shape.eval())
                v.set_value(np.zeros([0] * dim).astype(config.floatX))

    def _generate_ts(self, a_data, a_get_w_emb_i, a_get_c_emb_i):
        """Generate training set.

        Args:
        a_data (2-tuple(list, dict)):
          input data (discourse relations and parses)
        a_get_w_emb_i (method):
          custom method for retrieving the word embedding index
        a_get_c_emb_i (method):
          custom method for retrieving the conn embedding index

        Returns:
        (tuple(list, list)):
          lists of input features and expected classes

        """
        x, y = [], []
        if a_data is None:
            return (x, y)
        # generate features
        rels, parses = a_data
        # frequency of words in the corpus
        self._compute_w_stat(parses)
        for i, irel in rels:
            x.append((i, self._rel2x(irel, parses,
                                     a_get_w_emb_i,
                                     a_get_c_emb_i)))
            y.append(floatX(irel[SENSE]))
        return (x, y)

    def _rel2x(self, a_rel, a_parses, a_get_w_emb_i, a_get_c_emb_i):
        """Convert input relation to embeddings.

        Args:
        a_rel (dict):
          discourse relation whose tokens should be obtained
        a_parses (dict):
          parsed sentences
        a_get_w_emb_i (method):
          custom method for retrieving the word embedding index
        a_get_c_emb_i (method):
          custom method for retrieving the conn embedding index

        Returns:
        (np.array, np.array, np.array):
        embeddings of arg1, arg2, and connective

        """
        # print("rel2x: a_rel =", repr(a_rel))
        emb1 = self._arg2emb_idx(a_parses, a_rel, ARG1, a_get_w_emb_i)
        emb2 = self._arg2emb_idx(a_parses, a_rel, ARG2, a_get_w_emb_i)
        # print("emb_2 =", repr(emb2))
        conn_toks = a_rel[CONNECTIVE][RAW_TEXT]
        # print("conn_toks =", repr(conn_toks))
        emb_conn = a_get_c_emb_i(conn_toks)
        # print("emb_conn =", repr(emb_conn))
        return (emb1, emb2, emb_conn)

    def _arg2emb_idx(self, a_parses, a_rel, a_arg, a_get_emb_i):
        """Extract classification features for a given relation.

        Args:
        a_parses (dict):
          parsed sentences
        a_rel (dict):
          discourse relation whose tokens should be obtained
        a_arg (str):
          relation argument to obtain senses for
        a_get_emb_i (method):
          custom method for retrieving the embedding index

        Returns:
        (np.array):
        convert input relation to word embedding matrix

        """
        toks = [t[0] for t in
                self._get_toks_pos(a_parses[a_rel[DOC_ID]][SENTENCES],
                                   a_rel, a_arg)]
        if a_get_emb_i == self._get_test_w2v_emb_i or \
           a_get_emb_i == self._get_test_w2v_lstsq_emb_i:
            return np.asarray([a_get_emb_i(t) for t in toks])
        return np.asarray([a_get_emb_i(t) for t in toks], dtype="int32")

    def _init_w_emb(self):
        """Initialize task-specific word embeddings.

        Args:
        (void)

        Returns:
        (void)

        """
        self.W_EMB = theano.shared(
            value=HE_UNIFORM((self.w_i, self.ndim)), name="W_EMB")
        self._params.append(self.W_EMB)

    def _init_w2v_emb(self):
        """Initialize task-specific word embeddings.

        Args:
        (void)

        Returns:
        (void)

        """
        w_emb = np.empty((self.w_i, self.ndim))
        for w, i in self.w2emb_i.iteritems():
            if i == self.unk_w_i:
                continue
            w_emb[i] = self.w2v[w]
        self.W_EMB = theano.shared(value=floatX(w_emb),
                                   name="W_EMB")

    def _init_w2emb(self):
        """Compute a mapping from Word2Vec to embeddings.

        Args:
        (void)

        Returns:
        (void):
        modifies instance variable in place

        """
        # construct two matrices - one with the original word2vec
        # representations and another one with task-specific embeddings
        m = len(self.w2emb_i)
        n = self.ndim
        j = len(self._aux_keys)
        w2v_emb = floatX(np.empty((m, self.w2v.ndim)))
        task_emb = floatX(np.empty((m, n)))
        k = 0
        for w, i in self.w2emb_i.iteritems():
            k = i - j
            w2v_emb[k] = floatX(self.w2v[w])
            task_emb[k] = floatX(self.W_EMB[i])
        print("Computing task-specific transform matrix...", end="",
              file=sys.stderr)
        self.w2emb, res, rank, _ = np.linalg.lstsq(w2v_emb,
                                                   task_emb)
        self.w2emb = floatX(self.w2emb)
        print(" done (residuals: {:f}, w2v rank: {:d})".format(sum(res), rank),
              file=sys.stderr)

    def _get_train_w_emb_i(self, a_word):
        """Obtain embedding index for the given word.

        Args:
        a_word (str):
        word whose embedding index should be retrieved

        Returns:
        (int):
          embedding index od the given word

        """
        a_word = _norm_word(a_word)
        if a_word in self.w2emb_i:
            return self.w2emb_i[a_word]
        elif self._w_stat[a_word] < 2 and UNK_PROB():
            self.w2emb_i[a_word] = self.unk_w_i
            return self.unk_w_i
        else:
            i = self.w2emb_i[a_word] = self.w_i
            self.w_i += 1
            return i

    def _get_test_w_emb_i(self, a_word):
        """Obtain embedding index for the given word.

        Args:
        a_word (str):
          word whose embedding index should be retrieved

        Returns:
        (int):
          embedding index od the given word

        """
        a_word = _norm_word(a_word)
        return self.w2emb_i.get(a_word, self.unk_w_i)

    def _get_train_w2v_emb_i(self, a_word):
        """Obtain embedding index for the given word.

        Args:
        a_word (str):
          word whose embedding index should be retrieved

        Returns:
        (int):
          embedding index of the given word

        """
        a_word = _norm_word(a_word)
        if a_word in self.w2emb_i:
            return self.w2emb_i[a_word]
        elif a_word in self.w2v:
            i = self.w2emb_i[a_word] = self.w_i
            self.w_i += 1
            return i
        else:
            return self.unk_w_i

    def _get_test_w2v_emb_i(self, a_word):
        """Obtain embedding index for the given word.

        Args:
        a_word (str):
        word whose embedding index should be retrieved

        Returns:
        (np.array):
          embedding of the input word

        """
        a_word = _norm_word(a_word)
        emb_i = self.w2emb_i.get(a_word)
        if emb_i is None:
            if a_word in self.w2v:
                return floatX(self.w2v[a_word])
            return self.W_EMB[self.unk_w_i]
        return self.W_EMB[emb_i]

    def _get_test_w2v_lstsq_emb_i(self, a_word):
        """Obtain embedding index for the given word.

        Args:
        a_word (str):
        word whose embedding index should be retrieved

        Returns:
        (np.array):
          embedding of the input word

        """
        a_word = _norm_word(a_word)
        emb_i = self.w2emb_i.get(a_word)
        if emb_i is None:
            if a_word in self.w2v:
                return floatX(np.dot(self.w2v[a_word], self.w2emb))
            return self.W_EMB[self.unk_w_i]
        return self.W_EMB[emb_i]

    def _init_conn_emb(self):
        """Initialize task-specific connective embeddings.

        Args:
        (void)

        Returns:
        (void)

        """
        self.CONN_EMB = theano.shared(
            value=HE_UNIFORM((self.c_i, self.intm_dim)),
            name="CONN_EMB")
        self._params.append(self.CONN_EMB)

    def get_train_c_emb_i(self, a_conn):
        """Obtain embedding index for the given connective.

        Args:
        a_conn (str):
        connective whose embedding index should be retrieved

        Returns:
        (int):
        embedding index of the given connective

        """
        a_conn, _ = CHM.map_raw_connective(a_conn.lower())
        if a_conn in self.c2emb_i:
            ret = self.c2emb_i[a_conn]
        else:
            i = self.c2emb_i[a_conn] = self.c_i
            self.c_i += 1
            ret = i
        return np.asarray(ret, dtype="int32")

    def get_test_c_emb_i(self, a_conn):
        """Obtain embedding index for the given connective.

        Args:
        a_conn (str):
        connective whose embedding index should be retrieved

        Returns:
        (int):
        embedding index of the given connective

        """
        a_conn, _ = CHM.map_raw_connective(a_conn.lower())
        if a_conn in self.c2emb_i:
            ret = self.c2emb_i[a_conn]
        else:
            ret = self.unk_c_i
        return np.asarray(ret, dtype="int32")

    def _init_dropout(self, a_input):
        """Create a dropout layer.

        Args:
          a_input (theano.vector): input layer

        Returns:
          theano.vector: dropout layer

        """
        # the dropout layer itself
        output = TT.switch(self.use_dropout,
                           a_input * (TRNG.binomial(a_input.shape, p=0.5, n=1,
                                                    dtype=a_input.dtype)),
                           a_input * floatX(0.5))
        return output

    def _init_funcs(self, a_grads=None):
        """Compile theano functions.

        Args:
        a_grads (theano.shared or None):
        gradients of the trainign function

        Returns:
        (void):
        modifies instance variables in place

        """
        if a_grads:
            self._grad_shared, self._update, \
                self._rms_params = rmsprop(self._params, a_grads,
                                           [self.W_INDICES_ARG1,
                                            self.W_INDICES_ARG2,
                                            self.CONN_INDEX],
                                           self.Y_gold,
                                           self._cost)
        if self._compute_cost is None:
            self._compute_cost = theano.function([self.W_INDICES_ARG1,
                                                  self.W_INDICES_ARG2,
                                                  self.CONN_INDEX,
                                                  self.Y_gold], self._cost,
                                                 name="_compute_cost")
        # initialize prediction function
        if self._predict_func is None:
            self._predict_func = theano.function([self.W_INDICES_ARG1,
                                                  self.W_INDICES_ARG2,
                                                  self.CONN_INDEX],
                                                 self.Y_pred,
                                                 name="_predict_func")
            self._predict_func_emb = theano.function([self.EMB_ARG1,
                                                      self.EMB_ARG2,
                                                      self.CONN_INDEX],
                                                     self.Y_pred,
                                                     name="_predict_func_emb")
        # initialize debug function
        if self._debug_nn is None:
            self._debug_nn = theano.function([self.W_INDICES_ARG1,
                                              self.W_INDICES_ARG2,
                                              self.CONN_INDEX],
                                             [self.Y_pred],
                                             name="_debug_nn")

    def _init_wemb_funcs(self):
        """Initialize functions for obtaining word embeddings.

        Args:
        (void):

        Returns:
        (void):

        """
        if self._plain_w2v:
            if self.w2v is None:
                self.w2v = Word2Vec
            self.w2v.load()
            self.ndim = self.w2v.ndim
            self.get_train_w_emb_i = self._get_train_w2v_emb_i
            self.init_w_emb = self._init_w2v_emb
            if self._trained:
                self.get_test_w_emb_i = self._get_test_w2v_emb_i
                self._predict_func = self._predict_func_emb
            else:
                self.get_test_w_emb_i = self._get_train_w2v_emb_i
        elif self.lstsq:
            self.ndim = DFLT_VDIM
            self.get_train_w_emb_i = self._get_train_w2v_emb_i
            self.init_w_emb = self._init_w_emb
            if self._trained:
                if self.w2v is None:
                    self.w2v = Word2Vec
                    self.w2v.load()
                self.get_test_w_emb_i = self._get_test_w2v_lstsq_emb_i
                self._predict_func = self._predict_func_emb
            else:
                self.get_test_w_emb_i = self._get_train_w2v_emb_i
        else:
            # checked
            self.ndim = DFLT_VDIM
            self.get_train_w_emb_i = self._get_train_w_emb_i
            self.get_test_w_emb_i = self._get_test_w_emb_i
            self.init_w_emb = self._init_w_emb

    def _reset_funcs(self):
        """Set all compiled theano functions to None.

        Args:
        (void):

        Returns:
        (void):
        modifies instance variables in place

        """
        self._grad_shared = None
        self._update = None
        self._compute_cost = None
        self._predict_func = None
        self._debug_nn = None
        self.get_train_w_emb_i = None
        self.get_test_w_emb_i = None
        self.init_w_emb = None

    def _compute_w_stat(self, a_parses):
        """Compute word frequencies on the corpus.

        Args:
        a_parses (dict):
        CoNLL parses

        Returns:
        (void):
        modifies instance variables in place

        """
        self._w_stat = Counter(_norm_word(w[TOK_IDX])
                               for doc in a_parses.itervalues()
                               for sent in doc[SENTENCES]
                               for w in sent[WORDS])
