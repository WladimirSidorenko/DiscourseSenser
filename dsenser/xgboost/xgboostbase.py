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
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import f1_score, make_scorer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


##################################################################
# Constants
MAX_DEPTH = 9                   # maximim depth of tree
NTREES = 600                    # number of tree estimators
ALPHA = 0.05                    # learning rate
BASE_PARAM_GRID = {"clf__max_depth": [3 * i for i in xrange(1, 10)],
                   "clf__n_estimators": [100 * i for i in xrange(1, 10)]}


##################################################################
# Class
class XGBoostBaseSenser(object):
    """Subclass of explicit WangSenser using XGBoost

    """

    def __init__(self, a_clf=None, a_grid_search=False):
        """Class constructor.

        Args:
          a_clf (classifier or None):
            classifier to use or None for default
          a_grid_search (bool): use grid search for estimating
            hyper-parameters

        """
        classifier = a_clf
        self._gs = a_grid_search
        if a_clf is None:
            classifier = XGBClassifier(max_depth=MAX_DEPTH,
                                       n_estimators=NTREES,
                                       learning_rate=ALPHA,
                                       objective="multi:softprob")
            self._clf = classifier
        self._model = Pipeline([("vect", DictVectorizer()),
                                ("clf", classifier)])

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
