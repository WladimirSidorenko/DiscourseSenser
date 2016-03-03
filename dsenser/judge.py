#!/usr//bin/env python
# -*- coding: utf-8; mode: python; -*-

"""Module providing class for meta-classification.

Attributes:
Judge (class):
  class for joining decisions of single classifiers

"""

##################################################################
# Imports
from __future__ import absolute_import, print_function

import numpy as np

##################################################################
# Variables and Constants


##################################################################
# Class
class Judge(object):
    """Meta-classifier.

    This classifier unites decisions of other multiple independent classifiers.

    Attrs:

    Methods:

    """

    def __init__(self, a_n_x, a_n_y):
        """Class constructor.

        Args:
        a_n_x (int):
          number of underlying cassifiers
        a_n_y (int):
          number of classes to predict


        """
        self.n_x = a_n_x
        self.n_y = a_n_y

    def train(self, a_models, a_train_data, a_dev_data=None):
        """Method for training the model.

        Args:
        a_models (list(BaseSenser)):
          list of prediction models
        a_train_data (2-tuple(dict, dict)):
          list of training JSON data
        a_dev_data (2-tuple(dict, dict) or None):
          list of development JSON data

        Returns:
        (void)

        """
        pass

    def predict(self, a_models, a_rel, a_test_data):
        """Method for predicting sense of single relation.

        Args:
        a_models (list(BaseSenser)):
          list of prediction models
        a_rel (dict):
          discourse relation whose sense should be predicted
        a_test_data (2-tuple(dict, dict)):
          list of input JSON data

        Returns:
        str:
          most probable sense of discourse relation

        """
        return np.argmax(np.average([imodel.predict(a_rel,
                                                    a_test_data)
                                     for imodel in a_models], axis=0))
