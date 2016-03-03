#!/usr//bin/env python
# -*- coding: utf-8; mode: python; -*-

"""Module providing class for Wang sense disambiguation.

Attributes:
WangSenser (class):
  class that always chooses majority category  for sense disambiguation

"""

##################################################################
# Imports
from __future__ import absolute_import, print_function

from collections import defaultdict, Counter

from dsenser.base import BaseSenser
from dsenser.constants import CONNECTIVE, RAW_TEXT, SENSE

import numpy as np

##################################################################
# Variables and Constants


##################################################################
# Class
class WangSenser(BaseSenser):
    """Class using Wang classification for disambiguating connectives.

    Attrs:
    explicit (ImplicitSenser): classifier for implicit discourse relations
    implicit (ExplicitSenser): classifier
    n_y (int): number of distinct classes

    Methods:

    """

    def __init__(self):
        """Class constructor.

        Args:

        """
        self.dflt_sense = None
        self.conn2sense = {}
        self.n_y = -1

    def train(self, a_train_data, a_dev_data=None, a_n_y=-1):
        """Method for training the model.

        Args:
        a_train_data (2-tuple(dict, dict)):
          list of training JSON data
        a_dev_data (2-tuple(dict, dict) or None):
          list of development JSON data
        a_n_y (int):
          number of distinct classes

        Returns:
        (void)

        """
        self.n_y = a_n_y
        iconn = ""
        conn2sense = defaultdict(Counter)
        for irel in a_train_data[0]:
            iconn = self._normalize_conn(irel[CONNECTIVE][RAW_TEXT])
            for i, j in enumerate(irel[SENSE]):
                if j:
                    conn2sense[iconn][i] += 1
        # compute the most frequent sense and assign it to the default
        # connective
        all_senses = defaultdict(Counter)
        for istat in conn2sense.itervalues():
            all_senses.update(istat)
        # obtain the most frequent sense and use it for missing connectives
        self.dflt_sense = self._get_most_frequent_sense(all_senses)
        # for other connectives use their the respective most frequent sense
        for iconn, istat in conn2sense.iteritems():
            self.conn2sense[iconn] = self._get_most_frequent_sense(istat)

    def predict(self, a_rel, a_test_data):
        """Method for predicting sense of single relation.

        Args:
        a_rel (dict):
          discourse relation whose sense should be predicted
        a_test_data (2-tuple(dict, dict)):
          list of input JSON data

        Returns:
        str:
          most probable sense of discourse relation

        """
        iconn = self._normalize_conn(a_rel[CONNECTIVE][RAW_TEXT])
        isense = self.conn2sense.get(iconn, self.dflt_sense)
        ret = np.zeros(self.n_y)
        ret[isense] = 1.
        # print("major ret =", repr(ret))
        return ret

    def _get_most_frequent_sense(self, a_stat):
        """Obtain most frequent sense from statistcs.

        Args:
        a_stat (dict(str -> int)):
          statistics on senses

        Returns:
        str:
          most frequent sense in statistics

        """
        if not a_stat:
            raise RuntimeError("Empty statistics passed as argument.")
        max_cnt = -1
        dflt_sense = None
        for isense, icnt in a_stat.iteritems():
            if icnt > max_cnt:
                max_cnt = icnt
                dflt_sense = isense
        return dflt_sense

    def _free(self):
        """Free resources used by the model.

        Args:
        (void):

        Returns:
        (void):

        """
        del self.dflt_sense
        del self.conn2sense
