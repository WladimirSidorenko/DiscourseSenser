#!/usr/bin/env python
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
from dsenser.utils import timeit

import numpy as np
import sys

##################################################################
# Variables and Constants


##################################################################
# Class
class MajorSenser(BaseSenser):
    """Class making majority predictions for connective senses.

    Attributes:
      dflt_sense (np.array or None): probabilities of all senses in
                         the corpus
      conn2sense (dict): mapping from connective to the conditional
                         probabilities of its senses
      n_y (int): number of distinct classes

    """

    def __init__(self):
        """Class constructor.

        """
        self.dflt_sense = None
        self.conn2sense = {}
        self.n_y = -1

    @timeit("Training majority class classifier...")
    def train(self, a_train_data, a_dev_data=None, a_n_y=-1,
              a_i=-1, a_train_out=None, a_dev_out=None):
        """Method for training the model.

        Args:
          a_train_data (tuple(list, dict)):
            training JSON data
          a_dev_data (tuple(list, dict) or None):
            development JSON data
          a_n_y (int):
            number of distinct classes
          a_i (int):
            row index for the output predictions
          a_train_out (numpy.array or None):
            predictions for the training set
          a_dev_out (numpy.array or None):
            predictions for the development set

        Returns:
          void:

        Note:
          updates ``a_train_out`` and ``a_dev_out`` in place if not None

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
        self.dflt_sense = self._get_sense_stat(all_senses)
        # for other connectives use their the respective most frequent sense
        for iconn, istat in conn2sense.iteritems():
            self.conn2sense[iconn] = self._get_sense_stat(istat)
        # make predictions for the training and development set instances if
        # needed
        if a_i >= 0:
            if a_train_out is not None:
                for i, irel in enumerate(a_train_data[0]):
                    self.predict(irel, a_train_data, a_train_out[i], a_i)
            if a_dev_data is not None:
                for i, irel in enumerate(a_dev_data[0]):
                    self.predict(irel, a_dev_data, a_dev_out[i], a_i)

    def predict(self, a_rel, a_data, a_ret, a_i):
        """Method for predicting sense of single relation.

        Args:
          a_rel (dict):
            discourse relation whose sense need to be predicted
          a_data (2-tuple(dict, dict)):
            list of input JSON data
          a_ret (np.array):
            prediction matrix
          a_i (int):
            row index in the output vector

        Returns:
          void:

        Note:
          updates ``a_ret[a_i]`` in place

        """
        iconn = self._normalize_conn(a_rel[CONNECTIVE][RAW_TEXT])
        isense = self.conn2sense.get(iconn, self.dflt_sense)
        for i in xrange(len(isense)):
            a_ret[a_i][i] += isense[i]

    def _get_sense_stat(self, a_stat):
        """Generate sense statistcs.

        Args:
          a_stat (dict): statistics on senses

        Returns:
          (np.array): prior probabilities of senses

        """
        ret = np.zeros(self.n_y)
        for i, j in a_stat.iteritems():
            ret[i] = j
        ret /= float(sum(ret))
        return ret

    def _free(self):
        """Free resources used by the model.

        """
        del self.dflt_sense
        del self.conn2sense
        self.n_y = -1
