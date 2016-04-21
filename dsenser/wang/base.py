#!/usr//bin/env python
# -*- coding: utf-8; mode: python; -*-

"""Module providing abstract interface class for Wang sense classification.

Attributes:
WangBaseSenser (class):
  abstract class defining interface for explicit and implicit classifier

"""

##################################################################
# Imports
from __future__ import absolute_import, print_function

from dsenser.base import BaseSenser
from dsenser.constants import SENSE

import abc
import numpy as np


##################################################################
# Class
class WangBaseSenser(BaseSenser):
    """Abstract class for disambiguating relation senses.

    Attrs:
    n_y (int): number of distinct classes

    Methods:

    """
    __metaclass__ = abc.ABCMeta

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
        x_train, y_train = self._generate_ts(a_train_data)
        x_dev, y_dev = self._generate_ts(a_dev_data)
        # fit the model
        self._model.fit([el[-1] for el in x_train], y_train)
        if a_i >= 0:
            if a_train_out is not None:
                for i, x_i in x_train:
                    self._predict(x_i, a_train_out[i], a_i)
            if a_dev_out is not None:
                for i, x_i in x_dev:
                    self._predict(x_i, a_dev_out[i], a_i)

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
        feats = self._extract_features(a_rel, a_data[-1])
        self._predict(feats, a_ret, a_i)

    @abc.abstractmethod
    def _extract_features(self, a_rel, a_parses):
        """Extract classification features for a given relation.

        Args:
        a_rel (dict):
          discourse relation to extract features for
        a_parses (dict):
          parsed sentences

        Returns:
        (void):

        """
        raise NotImplementedError

    def _predict(self, a_feats, a_ret, a_i):
        """Method for predicting sense of single relation.

        Args:
        a_feats (dict):
          features of the input instance
        a_ret (np.array):
          output prediction vector
        a_i (int):
          row index in the output vector

        Returns:
        (void):
          updates ``a_ret`` in place

        """
        # obtain model's estimates
        dec = self._model.decision_function(a_feats)
        if len(dec.shape) > 1:
            dec = np.mean(dec, axis=0)
        # map model's classes to original indices
        for i, ival in enumerate(dec):
            a_ret[a_i][self._model.classes_[i]] = ival
        # normalize using softmax
        exp_ret = np.exp(a_ret[a_i])
        a_ret[a_i] = exp_ret / (sum(exp_ret) or 1e10)

    def _free(self):
        """Free resources used by the model.

        Args:
        (void):

        Returns:
        (void):

        """
        self.n_y = -1

    def _generate_ts(self, a_data):
        """Generate training set.

        Args:
        a_data (2-tuple(list, dict)):
          input data (discourse relations and parses)

        Returns:
        (tuple(list, list)):
          lists of input features and expected classes

        """
        x, y = [], []
        if a_data is None:
            return (x, y)
        x_i = y_i = None
        # generate features
        for i, irel in a_data[0]:
            x_i = self._extract_features(irel, a_data[1])
            if not x_i:
                continue
            x.append((i, x_i))
            y_i = np.argmax(irel[SENSE])
            y.append(y_i)
        return (x, y)
