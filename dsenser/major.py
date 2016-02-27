#!/usr//bin/env python
# -*- coding: utf-8; mode: python; -*-

"""Module providing class for majority class sense disambiguation.

Attributes:
Major (class):
  class that always chooses majority category  for sense disambiguation

"""

##################################################################
# Imports
from __future__ import absolute_import, print_function

from collections import defaultdict, Counter

from dsenser.base import BaseSenser
from dsenser.constants import CONNECTIVE, RAW_TEXT, SENSE


##################################################################
# Variables and Constants


##################################################################
# Class
class MajorSenser(BaseSenser):
    """Class making majority predictions for connective senses.

    Attrs:
    dflt_sense (str or None): most frequent sense in corpus
    conn2sense (dict): mapping from connective to its most frequent sense

    Methods:

    """

    def __init__(self):
        """Class constructor.

        Args:
        (void)


        """
        self.dflt_sense = None
        self.conn2sense = {}

    def train(self, a_train_data, a_path=None, a_dev_data=None):
        """Method for training the model.

        Args:
        a_train_data (2-tuple(dict, dict)):
          list of training JSON data
        a_path (str or None):
          path for model to be stored
        a_dev_data (2-tuple(dict, dict) or None):
          list of development JSON data

        Returns:
        (void)

        """
        conn2sense = defaultdict(Counter)
        for irel in a_train_data[0]:
            conn2sense[irel[CONNECTIVE][RAW_TEXT].lower()].update(
                irel[SENSE])
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
        # store model
        self._dump(a_path)

    def _predict(self, a_rel, a_test_data):
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
        return self.conn2sense.get(
            a_rel[CONNECTIVE][RAW_TEXT].lower(), self.dflt_sense)

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
