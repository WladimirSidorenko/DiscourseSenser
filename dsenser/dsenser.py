#!/usr/bin/env python
# -*- coding: utf-8; mode: python; -*-

"""Module providing class for sense disambiguation of connectives.

Attributes:
DiscourseSenser (class): class for sense disambiguation of connectives

"""

##################################################################
# Imports
from __future__ import absolute_import, print_function, unicode_literals

from dsenser.constants import ARG1, ARG2, CHAR_SPAN, CONNECTIVE, ENCODING, \
    RAW_TEXT, SENSE, TOK_LIST, TOK_OFFS_IDX, TYPE, DFLT_MODEL_PATH, \
    DFLT_MODEL_TYPE, DFLT_ECONN_PATH, ALT_LEX, EXPLICIT, IMPLICIT, SVD, \
    LSTM, MJR, WANG, XGBOOST, PARSE_TREE, DEPS, WORDS, SENTENCES, SHORT2FULL
from dsenser.utils import timeit

from collections import Iterable
from cPickle import dump, load

import codecs
import gc
import numpy as np
import os
import sys

##################################################################
# Variables and Constants
# load default explicit discourse connectives
DFLT_CONN = set(["upon"])
with codecs.open(DFLT_ECONN_PATH, 'r', ENCODING) as ifile:
    for iline in ifile:
        iline = iline.strip()
        if iline:
            DFLT_CONN.add(iline)


##################################################################
# Class
class DiscourseSenser(object):
    """Class for sense disambiguation of connectives.

    Attributes:
    models (list(BaseSenser)):
      sense disambiguation models
    judge (dsenser.Judge):
      meta-classifier
    cls2idx (dict):
      mapping from class to index
    idx2cls (dict):
      mapping from index to class
    econn (set):
      connectives marking explicit relations

    """

    def __init__(self, a_model=None):
        """Class constructor.

        Args:
        a_model (str or None): path to serialized model or None

        """
        self.models = []
        self.model_paths = []
        self.judge = None
        self.cls2idx = {}
        self.idx2cls = {}
        self.wbench = None
        self.econn = set([self._normalize_conn(iconn) for iconn in DFLT_CONN])
        # load serialized model
        if a_model is not None:
            self._load(a_model)

    def train(self, a_train_data, a_type=DFLT_MODEL_TYPE,
              a_path=DFLT_MODEL_PATH, a_dev_data=None,
              a_w2v=False, a_lstsq=False):
        """Train specified model(s) on the provided data.

        Args:
        a_train_data (list or None):
          training set
        a_path (str):
          path for storing the model
        a_type (str):
          type of the model to be trained
        a_dev_data (list or None):
          development set
        a_w2v (bool):
          use word2vec embeddings
        a_lstsq (bool):
          use least squares method

        Returns:
          (void)

        """
        if a_type == 0:
            raise RuntimeError("No model type specified.")
        if a_dev_data is None:
            a_dev_data = ([], {})
        # initialize
        if a_type & MJR:
            from dsenser.major import MajorSenser
            self.models.append(MajorSenser())
        if a_type & WANG:
            from dsenser.wang import WangSenser
            self.models.append(WangSenser())
        if a_type & XGBOOST:
            from dsenser.xgboost import XGBoostSenser
            self.models.append(XGBoostSenser())
        # NN models have to go last, since we are pruning the parses for them
        # to free some memory
        nn_used = False
        if a_type & SVD:
            from dsenser.svd import SVDSenser
            # since we cannot differentiate SVD yet, we can only use word2vec
            # embeddings
            if not a_w2v or a_lstsq:
                print("SVD senser does not support task-specific embeddings "
                      "and least squares yet.", file=sys.stderr)
            self.models.append(SVDSenser(a_w2v=True, a_lstsq=False,
                                         a_max_iters=256))
            nn_used = True
        if a_type & LSTM:
            from dsenser.lstm import LSTMSenser
            self.models.append(LSTMSenser(a_w2v, a_lstsq))
            nn_used = True
        # convert classes to indices
        self._sense2idx(a_train_data[0])
        # train models and remember their predictions (temporarly commented due
        # to memory optimization, since we are not using the judge now)
        # x_train = np.zeros((len(a_train_data[0]), len(self.models),
        #                     len(self.cls2idx)))
        # x_dev = np.zeros((len(a_dev_data[0] if a_dev_data else ()),
        #                   len(self.models), len(self.cls2idx)))
        i = 0
        data_pruned = False
        imodel = x_train = x_dev = None
        imodel_name = imodel_path = ""
        while i < len(self.models):
            imodel = self.models[i]
            imodel_name = imodel.__class__.__name__
            imodel_path = os.path.relname(a_path + '.' + imodel_name,
                                          a_path)
            if nn_used and not data_pruned:
                from dsenser.svd import SVDSenser
                from dsenser.lstm import LSTMSenser
                if isinstance(imodel, LSTMSenser) or \
                   isinstance(imodel, SVDSenser):
                    a_train_data = self._prune_data(*a_train_data)
                    a_dev_data = self._prune_data(*a_dev_data)
                    data_pruned = True
            # i = -1 (means do not make predictions for the judge)
            # imodel.train(a_train_data, a_dev_data, len(self.cls2idx),
            #              i, x_train, x_dev)
            imodel.train(a_train_data, a_dev_data, len(self.cls2idx),
                         -1, x_train, x_dev)
            self._dump(imodel, imodel_path)
            self.model_paths.append(imodel_path)
            self.models[i] = imodel = None
            gc.collect()
            i += 1
        # convert training and development sets to the format appropriate for
        # the judge
        # x_train = [(x_i, irel, irel[SENSE])
        #            for x_i, irel in zip(x_train, a_train_data[0])]
        # x_dev = [(x_i, irel, irel[SENSE])
        #          for x_i, irel in zip(x_dev, a_dev_data[0])]
        # train the judge
        # from dsenser.judge import Judge
        # self.judge = Judge(len(self.models), len(self.cls2idx))
        # self.judge.train(x_train, x_dev)
        # dump model (clean the model list before)
        self.models = []
        self._dump(self, a_path)

    def predict(self, a_data):
        """Determine senses of discourse connectives.

        This is a memory-optimized version of prediction function.  Due to
        these optimizations, however, it does not support the judge model.

        Args:
        a_data (list):
          input data to be analyzed

        Returns:
        (void):
          updates input set in place

        """
        if not self.model_paths:
            raise RuntimeError(
                "No paths to trained models are provided to make predictions.")

        rels = a_data[0]
        # normalize input relations
        self._preprocess_rels(rels)
        # predict sense
        imodel = isense = None
        # allocate space for predictions
        self.wbench = np.zeros((len(rels), len(self.cls2idx)))
        # iterate over each trained model and sum up their predictions
        for ipath in self.model_paths:
            print("ipath =", repr(ipath), file=sys.stderr)
            with open(ipath, "rb") as ifile:
                imodel = load(ifile)
            for i, irel in enumerate(rels):
                imodel.predict(irel, a_data, self.wbench, i)
            del imodel
            imodel = None
            gc.collect()
        # make final judgements
        idx = -1
        isense = None
        for i, irel in enumerate(rels):
            idx = int(np.argmax(self.wbench[i]))
            isense = self.idx2cls[idx]
            irel[SENSE].append(SHORT2FULL.get(isense, isense))
        # free memory occupied by workbench
        del self.wbench
        self.wbench = None
        gc.collect()
        # postprocess input relations
        self._postprocess_rels(rels)

    def _predict(self, a_rel, a_data):
        """Determine sense of discourse relation.

        Args:
        a_rel (dict):
          JSON instance representing discourse relation
        a_data (list):
          2-tuple(dict, dict): input rels and parses

        Returns:
        (tuple(str, float)):
          predicted label and its probability

        """
        # the best performing strategy so far is to return the highest mean
        # judgment
        x = self._prejudge(a_rel, a_data)
        x_mean = np.mean(x, axis=0)
        idx = np.argmax(x_mean)
        lbl = self.idx2cls[int(idx)]
        return (lbl, x_mean[idx])
        # earlier we were using a pre-trained judge tensor
        # idx, iprob = self.judge.predict(a_rel, self._prejudge(a_rel, a_data))
        # lbl = self.idx2cls[int(idx)]
        # return (lbl, iprob)

    def _prejudge(self, a_rel, a_data):
        """Collect judgments of single classifiers.

        Args:
        a_rel (dict):
          discourse relation whose sense should be predicted
        a_data (2-tuple(dict, dict)):
          list of input JSON data

        Returns:
        (np.array):
          modified ``a_ret``

        """
        if self.wbench is None:
            self.wbench = np.zeros((len(self.models), len(self.cls2idx)))
        for i, imodel in enumerate(self.models):
            imodel.predict(a_rel, a_data, self.wbench, i)
        return self.wbench

    def _prune_data(self, a_rels, a_parses):
        """Remove unnecessary information from data.

        Args:
        a_rels (list):
          list of input discourse relations
        a_parses (dict):
          parse trees

        Returns:
        2-tuple(list, dict):
          abridged input data

        """
        arg = None
        # clean-up relations
        for irel in a_rels:
            irel.pop("ID")
            irel[CONNECTIVE].pop(CHAR_SPAN)
            arg = irel[ARG1]
            arg.pop(CHAR_SPAN)
            arg.pop(RAW_TEXT)
            arg = irel[ARG2]
            arg.pop(CHAR_SPAN)
            arg.pop(RAW_TEXT)
        # clean-up parses
        w_attrs = None
        for isentences in a_parses.itervalues():
            for isent in isentences[SENTENCES]:
                isent.pop(PARSE_TREE)
                isent.pop(DEPS)
                for iword in isent[WORDS]:
                    iword[-1].clear()
        return (a_rels, a_parses)

    def _preprocess_rels(self, a_rels):
        """Preprocess input relations.

        Args:
        a_rels (list):
          input relations to be preprocessed

        Returns:
        (void):
          modifies ``a_rels`` in place

        """
        arg1 = arg2 = None
        for irel in a_rels:
            arg1 = irel[ARG1]
            arg1.pop(CHAR_SPAN, None)
            arg1.pop(RAW_TEXT, None)

            arg2 = irel[ARG2]
            arg2.pop(CHAR_SPAN, None)
            arg2.pop(RAW_TEXT, None)

            if len(irel[CONNECTIVE][TOK_LIST]) == 0:
                irel[CONNECTIVE][RAW_TEXT] = ""
            if not SENSE in irel:
                irel[SENSE] = []
            if not TYPE in irel or not irel[TYPE]:
                irel[TYPE] = self._get_type(irel)

    def _postprocess_rels(self, a_rels):
        """Postprocess input relations.

        Args:
        a_rels (list):
          input relations to be preprocessed

        Returns:
        (void):
          modifies ``a_rels`` in place

        """
        arg1 = arg2 = None
        for irel in a_rels:
            arg1 = irel[ARG1]
            arg2 = irel[ARG2]

            irel[CONNECTIVE].pop(CHAR_SPAN, None)
            if irel[TYPE] != EXPLICIT:
                irel[CONNECTIVE].pop(RAW_TEXT, None)
            arg1[TOK_LIST] = self._normalize_tok_list(arg1[TOK_LIST])
            arg2[TOK_LIST] = self._normalize_tok_list(arg2[TOK_LIST])
            irel[CONNECTIVE][TOK_LIST] = self._normalize_tok_list(
                irel[CONNECTIVE][TOK_LIST])

    def _sense2idx(self, a_rels):
        """Convert symbolic senses to vectors.

        Args:
        a_rels (list):
        list of discourse relations

        Returns:
        (void):
        updates ``a_rels`` in place

        """
        n_senses = 0
        isense = isenses = None
        for irel in a_rels:
            isenses = irel[SENSE]
            for isense in isenses:
                isense = SHORT2FULL.get(isense, isense)
                if isense not in self.cls2idx:
                    self.cls2idx[isense] = n_senses
                    self.idx2cls[n_senses] = isense
                    n_senses += 1
            if irel[TYPE] == EXPLICIT:
                self.econn.add(self._normalize_conn(
                    irel[CONNECTIVE][RAW_TEXT]))
            else:
                irel[CONNECTIVE][RAW_TEXT] = ""
        vsense = None
        for irel in a_rels:
            isenses = irel[SENSE]
            vsense = np.zeros(n_senses)
            for isense in isenses:
                isense = SHORT2FULL.get(isense, isense)
                vsense[self.cls2idx[isense]] = 1
            irel[SENSE] = vsense / sum(vsense)

    def _get_type(self, a_rel):
        """Determine type of discourse relation.

        Args:
        a_rel (dict):
          relation in question

        Returns:
        (void)

        """
        conn = a_rel[CONNECTIVE]
        conn_txt = conn.get(RAW_TEXT, None)
        if conn_txt is not None:
            if not conn.get(TOK_LIST, None):
                rel = IMPLICIT
            elif self._normalize_conn(conn_txt) in self.econn:
                rel = EXPLICIT
            else:
                rel = ALT_LEX
        else:
            rel = IMPLICIT
        return rel

    def _normalize_tok_list(self, a_tok_list):
        """Flatten token list, only leaving doc offsets.

        Args:
        a_tok_list (list(list(int)):
          relation in question

        Returns:
        (void)

        """
        return [el[TOK_OFFS_IDX] if isinstance(el, Iterable) else el
                for el in a_tok_list]

    def _normalize_conn(self, a_conn):
        """Normalize connective form.

        Args:
        a_conn (str):
          connectve to be normalized

        Returns:
        (void)

        """
        return a_conn.strip().lower()

    def _dump(self, a_obj, a_path=None):
        """Dump this model to disc at the given path.

        Args:
        a_obj (object):
          object being dumped
        a_path (str or None):
          path to file in which to store the model

        Returns:
        (void)

        """
        if a_path is None:
            return
        # check directory
        dirname = os.path.dirname(a_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        elif not os.path.exists(a_path):
            if not os.access(dirname, os.W_OK) or \
               not os.path.isdir(dirname):
                raise RuntimeError("Cannot write to directory '{:s}'.".format(
                    dirname))
        else:
            if not os.access(a_path, os.W_OK):
                raise RuntimeError("Cannot write to file '{:s}'.".format(
                    a_path))
        with open(a_path, "wb") as ofile:
            dump(a_obj, ofile)

    def _load(self, a_path):
        """Load serialized model from disc.

        Args:
        a_path (str):
          path to file from which to load the model

        Returns:
        (void)

        """
        # load paths to serialized models
        with open(a_path, "rb") as ifile:
            self._move(load(ifile))
        bfname = os.path.dirname(a_path)
        # normalize paths to serialized models
        self.model_paths = [os.path.join(bfname, ipath)
                            for ipath in self.model_paths]

    def _move(self, a_senser):
        """Load serialized model from disc.

        Args:
        a_senser (dsenser.DiscourseSenser):
          source DiscourseSenser model

        Returns:
        (void)

        """
        self._free()
        self._copy(a_senser)

    def _copy(self, a_senser):
        """Load serialized model from disc.

        Args:
        a_senser (dsenser.DiscourseSenser):
          source DiscourseSenser model

        Returns:
        (void)

        """
        self.wbench = a_senser.wbench
        self.models = a_senser.models
        self.model_paths = a_senser.model_paths
        self.judge = a_senser.judge
        self.cls2idx = a_senser.cls2idx
        self.idx2cls = a_senser.idx2cls
        self.econn = a_senser.econn

    def _free(self):
        """Free allocated resources.

        Args:
        (void):

        Returns:
        (void)

        """
        for imodel in self.models:
            imodel._free()
        del self.models[:]
        del self.judge
        del self.cls2idx
        del self.idx2cls
        del self.econn
