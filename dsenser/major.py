#!/usr//bin/env python
# -*- coding: utf-8; mode: python; -*-

"""Module providing class for majority class sense disambiguation.

Attributes:
MajorSenser (class):
  class that always chooses majority category  for sense disambiguation

"""

##################################################################
# Imports
from __future__ import absolute_import, print_function

from collections import defaultdict, Counter

from dsenser.base import BaseSenser
from dsenser.constants import CONNECTIVE, RAW_TEXT, SENSE

import numpy as np
import sys

##################################################################
# Variables and Constants


##################################################################
# Class
class MajorSenser(BaseSenser):
    """Class making majority predictions for connective senses.

    Attrs:
    dflt_sense (str or None): most frequent sense in corpus
    conn2sense (dict): mapping from connective to its most frequent sense
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
        print("Training majority class classifier ...", end="",
              file=sys.stderr)
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
        self.dflt_sense = self._get_sense_stat(all_senses)
        # for other connectives use their the respective most frequent sense
        for iconn, istat in conn2sense.iteritems():
            self.conn2sense[iconn] = self._get_sense_stat(istat)
        print(" done", file=sys.stderr)

    def predict(self, a_rel, a_test_data):
        """Method for predicting sense of single relation.

        Args:
        a_rel (dict):
          discourse relation whose sense should be predicted
        a_test_data (2-tuple(dict, dict)):
          list of input JSON data

        Returns:
        (np.array):
          prior probabilities of senses

        """
        iconn = self._normalize_conn(a_rel[CONNECTIVE][RAW_TEXT])
        return self.conn2sense.get(iconn, self.dflt_sense)

    def _get_sense_stat(self, a_stat):
        """Generate sense statistcs.

        Args:
        a_stat (dict(str -> int)):
          statistics on senses

        Returns:
        (np.array):
          prior probabilities of senses

        """
        ret = np.zeros(self.n_y)
        for i, j in a_stat.iteritems():
            ret[i] = j
        ret /= float(sum(ret))
        return ret

    def _free(self):
        """Free resources used by the model.

        Args:
        (void):

        Returns:
        (void):

        """
        del self.dflt_sense
        del self.conn2sense
        self.n_y = -1
