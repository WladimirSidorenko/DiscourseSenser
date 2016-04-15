#!/usr//bin/env python
# -*- coding: utf-8; mode: python; -*-

"""Module providing abstract class for sense disambiguation.

Attributes:
BaseSenser (class):
  class that always chooses majority category  for sense disambiguation

"""

##################################################################
# Imports
from __future__ import absolute_import, print_function

from dsenser.constants import POS, SNT_ID, TOK_IDX, TOK_ID, TOK_LIST, WORDS
from dsenser.scorer.conn_head_mapper import ConnHeadMapper
from dsenser.resources import CHM
from dsenser.utils import is_explicit

from collections import defaultdict


##################################################################
# Variables and Constants

##################################################################
# Methods


##################################################################
# Class
class BaseSenser(object):
    """Abstract class for sense disambiguation of connectives.

    Attrs:

    Methods:
    train: pure abstract method

    """

    def __init__(self):
        """Class constructor.

        """
        self.explicit = None
        self.implicit = None

    def train(self, a_train_data, a_dev_data=None, a_n_y=-1,
              a_i=-1, a_train_out=None, a_dev_out=None):
        """Method for training the model.

        Args:
        a_train_data (2-tuple(list, dict)):
          list of gold relations and dict with parses
        a_dev_data (2-tuple(list, dict) or None):
          list of development relations and dict with parses
        a_n_y (int):
          number of distinct classes
        a_i (int):
          row index for the output predictions
        a_train_out (np.array or None):
          predictions for the training set
        a_dev_out (np.array or None):
          predictions for the training set

        Returns:
        (void)

        """
        self.n_y = a_n_y
        explicit_train, implicit_train = self._divide_ds(a_train_data)
        explicit_dev, implicit_dev = self._divide_ds(a_dev_data)
        self.implicit.train(implicit_train, implicit_dev, a_n_y,
                            a_i, a_train_out, a_dev_out)
        self.explicit.train(explicit_train, explicit_dev, a_n_y,
                            a_i, a_train_out, a_dev_out)

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
        str:
          most probable sense of discourse relation

        """
        if is_explicit(a_rel):
            return self.explicit.predict(a_rel, a_data, a_ret, a_i)
        return self.implicit.predict(a_rel, a_data, a_ret, a_i)

    def _free(self):
        """Free resources used by the model.

        Args:
        (void):

        Returns:
        (void):

        """
        del self.explicit
        del self.implicit
        self.n_y = -1

    def _divide_ds(self, a_ds):
        """Separate dataset into explicit and implicit instances.

        Args:
        a_ds (2-tuple(dict, dict)):
        list of gold relations and dict with parses

        Returns:
        (((list, dict), (list, dict))):
        trainings set with explicit and implicit connectives

        """
        if not a_ds:
            return (([], {}), ([], {}))
        explicit_instances = []
        implicit_instances = []
        for i, irel in enumerate(a_ds[0]):
            if is_explicit(irel):
                explicit_instances.append((i, irel))
            else:
                implicit_instances.append((i, irel))
        ret = ((explicit_instances, a_ds[1]),
               (implicit_instances, a_ds[1]))
        return ret

    def _normalize_conn(self, a_conn):
        """Normalize connective form.

        Args:
        a_conn (str):
          connectve to be normalized

        Returns:
        (void)

        """
        a_conn = a_conn.strip().lower()
        if a_conn:
            return CHM.map_raw_connective(a_conn)[0]
        return a_conn

    def _get_toks_pos(self, a_parses, a_rel, a_arg):
        """Method for getting raw tokens with their parts of speech.

        Args:
        a_parses (dict):
          parsed sentences
        a_rel (dict):
          discourse relation whose tokens should be obtained
        a_arg (str):
          relation argument to obtain senses for

        Returns:
        (list(tuple(str, str))):
          list of tokens and their parts of speech

        """
        ret = []
        snt = wrd = None
        for s_id, w_ids in \
                self._get_snt2tok(a_rel[a_arg][TOK_LIST]).iteritems():
            snt = a_parses[s_id][WORDS]
            for w_id in w_ids:
                wrd = snt[w_id]
                ret.append((wrd[TOK_IDX].lower(), wrd[1][POS]))
        return ret

    def _get_snt2tok(self, a_tok_list):
        """Generate mapping from sentence indices to token lists.

        Args:
        a_tok_list (list(tuple(int, int))):
          list of sentence and token indices pertaining to the argument

        Returns:
        defaultdict(set):
          mapping from sentence indices to token lists

        """
        snt2tok_pos = defaultdict(set)
        for el in a_tok_list:
            snt_id = el[SNT_ID]
            snt2tok_pos[snt_id].add(el[TOK_ID])
        return snt2tok_pos
