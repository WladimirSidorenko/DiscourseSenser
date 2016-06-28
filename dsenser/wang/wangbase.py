#!/usr/bin/env python
# -*- coding: utf-8; mode: python; -*-

"""Module providing abstract interface class for Wang sense classification.

Attributes:
  WangBaseSenser (class):
    abstract class defining interface for explicit and implicit classifier

"""

##################################################################
# Imports
from __future__ import absolute_import, print_function

from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import f1_score, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from dsenser.base import BaseSenser
from dsenser.constants import SENSE

import numpy as np
import abc
import sys

##################################################################
# Constants
NFOLDS = 5
DFLT_C = 0.02
DFLT_EXP_C = 0.07
DFLT_IMP_C = DFLT_C
DFLT_CLS_WGHT = None
DFLT_PARAMS = {"class_weight": DFLT_CLS_WGHT, "loss": "hinge",
               "penalty": "l1", "dual": True, "multi_class": "crammer_singer"}


##################################################################
# Class
class WangBaseSenser(BaseSenser):
    """Abstract class for disambiguating relation senses.

    Attributes:
      n_y (int): number of distinct classes

    """
    __metaclass__ = abc.ABCMeta

    # private members
    PARAM_GRID = {"clf__C": [float(i)/100. for i in xrange(1, 11)]}

    def __init__(self, a_clf=None, a_grid_search=False):
        """Class constructor.

        Initialize classifier.

        Args:
          a_clf (classifier or None):
            classifier to use or None for default
          a_grid_search (bool): use grid search for estimating hyper-parameters

        """
        classifier = a_clf or LinearSVC(C=DFLT_C,
                                        **DFLT_PARAMS)
        self._gs = a_grid_search
        self._model = Pipeline([("vect", DictVectorizer()),
                                ("clf", classifier)])

    def train(self, a_train_data, a_dev_data=None, a_n_y=-1,
              a_i=-1, a_train_out=None, a_dev_out=None):
        """Method for training the model.

        Args:
          a_train_data (tuple[list, dict]):
            list of training JSON data
          a_dev_data (tuple[list, dict] or None):
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
          void:

        Note:
          updates ``a_train_out`` and ``a_dev_out`` in place

        """
        self.n_y = a_n_y
        x_train, y_train = self._generate_ts(a_train_data)
        x_dev, y_dev = self._generate_ts(a_dev_data)
        # determine cross-validation and grid-search strategy and fit the model
        if self._gs:
            if a_dev_data is None or not a_dev_data[0]:
                cv = StratifiedKFold(y_train, n_folds=NFOLDS, shuffle=True)
            else:
                cv = self._devset_cv(y_train, len(y_dev), NFOLDS)
                x_train += x_dev
                y_train += y_dev
            scorer = make_scorer(f1_score, average="macro")
            self._model = GridSearchCV(self._model, self.PARAM_GRID,
                                       scoring=scorer,
                                       cv=cv, n_jobs=-1, verbose=1)
        self._model.fit([el[-1] for el in x_train], y_train)
        # output best hyper-parameters
        if self._gs:
            print("Best params:", repr(self._model.best_params_),
                  file=sys.stderr)
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
          void:

        Note:
          updates ``a_ret[a_i]`` in place

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
          void:

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
          void:
            updates ``a_ret[a_i]`` in place

        """
        # obtain model's estimates
        dec = self._model.decision_function(a_feats)
        if len(dec.shape) > 1:
            dec = np.mean(dec, axis=0)
        # normalize using softmax
        exp_ret = np.exp(sum(dec)) or 1e10
        dec /= exp_ret
        # map model's classes to original indices
        for i, ival in enumerate(dec):
            a_ret[a_i][self._model.classes_[i]] += ival

    def _free(self):
        """Free resources used by the model.

        """
        self.n_y = -1

    def _generate_ts(self, a_data):
        """Generate training set.

        Args:
          a_data (2-tuple(list, dict)):
            input data (discourse relations and parses)

        Returns:
          tuple(list, list):
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

    def _devset_cv(self, a_y_train, a_n_dev, a_n_folds):
        """Generate train-test split from training and development data.

        Args:
          a_y_train (list[int]):
            list of training instances' tags
          a_n_dev (int):
            number of devset instances
          a_n_folds (int):
            number of folds

        Returrns:
          list[tuple]: list of training/testing folds

        """
        folds = []
        n_train = len(a_y_train)
        dev_ids = [n_train + i for i in xrange(a_n_dev)]
        # create stratified K-folds over the training data
        skf = StratifiedKFold(a_y_train, a_n_folds)
        for train_ids, test_ids in skf:
            folds.append((train_ids,
                          np.concatenate((test_ids, dev_ids))))
        return folds
