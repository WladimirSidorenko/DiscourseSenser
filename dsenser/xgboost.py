#!/usr//bin/env python
# -*- coding: utf-8; mode: python; -*-

"""Module providing class for Wang sense disambiguation.

Attributes:
WangSenser (class):
  class for XGBoost sense classification of explicit and implicit relations

"""

##################################################################
# Imports
from __future__ import absolute_import, print_function

from dsenser.wang import WangSenser
from dsenser.wang.explicit import WangExplicitSenser
from dsenser.wang.implicit import WangImplicitSenser
from dsenser.utils import timeit

from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

##################################################################
# COnstants
MAX_DEPTH = 3                   # maximim depth of tree
NTREES = 300                    # number of tree estimators
ALPHA = 0.05                    # learning rate
EPS = 0.1


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

    def __init__(self, a_clf=None):
        """Class constructor.

        Args:
        a_clf (classifier or None):
        classifier to use or None for default

        """
        self.n_y = self.eps = -1
        self.one = 1. - EPS
        classifier = a_clf or XGBClassifier(max_depth=MAX_DEPTH,
                                            n_estimators=NTREES,
                                            learning_rate=ALPHA)
        self._model = Pipeline([('vectorizer', DictVectorizer()),
                                # ('var_filter', SelectKBest(chi2,
                                #                            k=1500)),
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
        (void):
          updates ``a_ret`` in place

        """
        # obtain model's estimates
        dec = self._model.predict(a_feats)[0]
        if self.eps < 0:
            self.eps = EPS / float(self.n_y)
        # map model's classes to original indices
        for i in xrange(self.n_y):
            if i == dec:
                a_ret[a_i][i] = self.one
            else:
                a_ret[a_i][i] = self.eps


class XGBoostExplicitSenser(XGBoostBaseSenser, WangExplicitSenser):
    """Subclass of explicit WangSenser using XGBoost

    """

    @timeit("Training explicit XGBoost classifier...")
    def train(self, *args, **kwargs):
        super(WangExplicitSenser, self).train(*args, **kwargs)


class XGBoostImplicitSenser(XGBoostBaseSenser, WangImplicitSenser):
    """Subclass of explicit WangSenser using XGBoost

    """

    @timeit("Training implicit XGBoost classifier...")
    def train(self, *args, **kwargs):
        super(WangImplicitSenser, self).train(*args, **kwargs)


class XGBoostSenser(WangSenser):
    """Class for XGBoost classification of discourse relations.

    Attrs:
    explicit (ImplicitSenser): classifier for implicit discourse relations
    implicit (ExplicitSenser): classifier for explicit discourse relations
    n_y (int): number of distinct classes

    Methods:

    """

    def __init__(self):
        """Class constructor.

        Args:

        """
        self.explicit = XGBoostExplicitSenser()
        self.implicit = XGBoostImplicitSenser()
        self.n_y = -1
