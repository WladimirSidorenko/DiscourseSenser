#!/usr/bin/env python
# -*- coding: utf-8; mode: python; -*-

"""Module providing abstract interface class for XGBoost sense calssification.

Attributes:
  XGBoostBaseSenser (class):
    abstract class defining interface for explicit and implicit classifier

"""

##################################################################
# Imports
from __future__ import absolute_import, print_function

from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


##################################################################
# Constants
MAX_DEPTH = 3                   # maximim depth of tree
NTREES = 300                    # number of tree estimators
ALPHA = 0.05                    # learning rate


##################################################################
# Class
class XGBoostBaseSenser(object):
    """Subclass of explicit WangSenser using XGBoost

    """

    def __init__(self, a_clf=None):
        """Class constructor.

        Args:
          a_clf (classifier or None):
            classifier to use or None for default

        """
        classifier = a_clf
        self._clf = None
        if a_clf is None:
            classifier = XGBClassifier(max_depth=MAX_DEPTH,
                                       n_estimators=NTREES,
                                       learning_rate=ALPHA,
                                       objective="multi:softprob")
            self._clf = classifier
        self._model = Pipeline([('vectorizer', DictVectorizer()),
                                ('classifier', classifier)])

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
          void:

        Note:
          updates ``a_ret`` in place

        """
        ret = self._model.predict_proba(a_feats)[0]
        if self._clf is None:
            a_ret[a_i] += ret
        else:
            for i, j in enumerate(ret):
                a_ret[a_i][self._clf._le.inverse_transform(i)] += j
