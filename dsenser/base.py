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

import abc


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
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def train(self, a_train_data, a_dev_data=None, a_n_y=-1):
        """Abstract method defining interface for model training.

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
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, a_rel, a_test_data):
        """Abstract method defining interface for model usage.

        Args:
        a_rel (dict):
          discourse relation whose sense should be predicted
        a_test_data (2-tuple(dict, dict)):
          list of input JSON data

        Returns:
        (void):
        updates test data in place

        """
        raise NotImplementedError

    @abc.abstractmethod
    def _free(self):
        """Free resources used by the model.

        Args:
        (void):

        Returns:
        (void):

        """
        pass

    def _normalize_conn(self, a_conn):
        """Normalize connective form.

        Args:
        a_conn (str):
          connectve to be normalized

        Returns:
        (void)

        """
        return a_conn.strip().lower()
