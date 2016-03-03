#!/usr//bin/env python
# -*- coding: utf-8; mode: python; -*-

"""Module providing class for sense disambiguation of connectives.

Attributes:
DiscourseSenser (class): class for sense disambiguation of connectives

"""

##################################################################
# Imports
from __future__ import absolute_import, print_function, unicode_literals

from dsenser.constants import ARG1, ARG2, CHAR_SPAN, CONNECTIVE, RAW_TEXT, \
    SENSE, TOK_LIST, TOK_OFFS_IDX, TYPE, DFLT_MODEL_PATH, DFLT_MODEL_TYPE, \
    ALT_LEX, EXPLICIT, IMPLICIT, FFNN, LSTM, MJR, SVM, WANG
from dsenser.judge import Judge

from collections import Iterable
from cPickle import dump, load

import numpy as np
import os

##################################################################
# Variables and Constants


##################################################################
# Methods
def _flatten(a_list):
    """Flatten nested list.

    Args:
    a_list (list):
      possible nested list

    Returns:
    list:
     flat variant of the input list

    """
    for el in a_list:
        if isinstance(el, Iterable) and not isinstance(el, basestring):
            for sub in _flatten(el):
                yield sub
        else:
            yield el


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
        self.judge = None
        self.cls2idx = {}
        self.idx2cls = {}
        self.econn = set()
        # load serialized model
        if a_model is not None:
            self._load(a_model)

    def train(self, a_train_data, a_type=DFLT_MODEL_TYPE,
              a_path=DFLT_MODEL_PATH, a_dev_data=None):
        """Train specified model(s) on the provided data.

        Args:
        a_train_data (list or None):
          development set
        a_path (str):
          path for storing the model
        a_type (str):
          type of the model to be trained
        a_dev_data (list or None):
          development set

        Returns:
          (void)

        """
        if a_type == 0:
            raise RuntimeError("No model type specified.")
        # initialize
        if a_type & MJR:
            from dsenser.major import MajorSenser
            self.models.append(MajorSenser())
        # convert classes to indices
        n_senses = 0
        isense = isenses = None
        for irel in a_train_data[0]:
            isenses = irel[SENSE]
            for isense in isenses:
                if isense not in self.cls2idx:
                    self.cls2idx[isense] = n_senses
                    self.idx2cls[n_senses] = isense
                    n_senses += 1
            if irel[TYPE] == EXPLICIT:
                self.econn.add(self._normalize_conn(
                    irel[CONNECTIVE][RAW_TEXT]))
        vsense = None
        for irel in a_train_data[0]:
            isenses = irel[SENSE]
            vsense = np.zeros(n_senses)
            for isense in isenses:
                vsense[self.cls2idx[isense]] = 1
            irel[SENSE] = vsense / sum(vsense)
        # train models
        for imodel in self.models:
            imodel.train(a_train_data, a_dev_data, len(self.cls2idx))
        # train judge
        self.judge = Judge(len(self.models), len(self.cls2idx))
        self.judge.train(self.models, a_train_data, a_dev_data)
        # dump model
        self._dump(a_path)

    def predict(self, a_data):
        """Determine senses of discourse connectives.

        Args:
        a_data (list):
          input data to be analyzed

        Returns:
        (void):
          updates input set in place

        """
        if not self.models:
            raise RuntimeError(
                "No trained models are provided to make predictions.")
        arg1 = arg2 = None
        for irel in a_data[0]:
            arg1 = irel[ARG1]
            arg1.pop(CHAR_SPAN, None)
            arg1.pop(RAW_TEXT, None)
            arg1[TOK_LIST] = self._normalize_tok_list(arg1[TOK_LIST])

            arg2 = irel[ARG2]
            arg2.pop(CHAR_SPAN, None)
            arg2.pop(RAW_TEXT, None)
            arg2[TOK_LIST] = self._normalize_tok_list(arg2[TOK_LIST])

            if not SENSE in irel:
                irel[SENSE] = []
            irel[SENSE].append(self._predict(irel, a_data))
            if not TYPE in irel or not irel[TYPE]:
                irel[TYPE] = self._get_type(irel)
            irel[CONNECTIVE].pop(CHAR_SPAN, None)
            if irel[TYPE] != EXPLICIT:
                irel[CONNECTIVE].pop(RAW_TEXT, None)
            irel[CONNECTIVE][TOK_LIST] = self._normalize_tok_list(
                irel[CONNECTIVE][TOK_LIST])

    def _predict(self, a_rel, a_data):
        """Determine sense of discourse relation.

        Args:
        a_rel (dict):
          JSON instance representing discourse relation
        a_data (list):
          input data to be analyzed

        Returns:
        (void):
          updates input set in place

        """
        return self.idx2cls[self.judge.predict(self.models, a_rel, a_data)]

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

    def _dump(self, a_path=None):
        """Dump this model to disc at the given path.

        Args:
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
            dump(self, ofile)

    def _load(self, a_path):
        """Load serialized model from disc.

        Args:
        a_path (str):
          path to file from which to load the model

        Returns:
        (void)

        """
        with open(a_path, "rb") as ifile:
            self._move(load(ifile))

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
        self.models = a_senser.models
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
