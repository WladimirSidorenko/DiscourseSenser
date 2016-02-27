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

from dsenser.constants import ARG1, ARG2, CHAR_SPAN, CONNECTIVE, \
    RAW_TEXT, SENSE, TOK_LIST, TYPE, TOK_OFFS_IDX, EXPLICIT, IMPLICIT

from collections import Iterable
from cPickle import dump
import abc
import os


##################################################################
# Methods
def _flatten(a_list):
    """Flatten nested list.

    Args:
    a_list (list):
      possible nested list

    Returns:
    list:
     flat variant of the input list

    """
    for el in a_list:
        if isinstance(el, Iterable) and not isinstance(el, basestring):
            for sub in _flatten(el):
                yield sub
        else:
            yield el


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
    def train(self, a_train_data, a_path=None, a_dev_data=None):
        """Abstract method defining interface for model training.

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
        raise NotImplementedError

    def predict(self, a_test_data):
        """Abstract method defining interface for model testing.

        Args:
        a_test_data (2-tuple(dict, dict)):
          list of input JSON data

        Returns:
        (void):
        updates test data in place

        """
        arg1 = arg2 = None
        for irel in a_test_data[0]:
            arg1 = irel[ARG1]
            arg1.pop(CHAR_SPAN, None)
            arg1.pop(RAW_TEXT, None)
            arg1[TOK_LIST] = self._normalize_tok_list(arg1[TOK_LIST])

            arg2 = irel[ARG2]
            arg2.pop(CHAR_SPAN, None)
            arg2.pop(RAW_TEXT, None)
            arg2[TOK_LIST] = self._normalize_tok_list(arg2[TOK_LIST])

            if not SENSE in irel:
                irel[SENSE] = []
            irel[SENSE].append(self._predict(irel, a_test_data))
            if not TYPE in irel or not irel[TYPE]:
                irel[TYPE] = self._get_type(irel)
            irel[CONNECTIVE][TOK_LIST] = self._normalize_tok_list(
                irel[CONNECTIVE][TOK_LIST])

    @abc.abstractmethod
    def _predict(self, a_rel, a_test_data):
        """Abstract method defining interface for model testing.

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

    def _get_type(self, a_rel):
        """Determine type of discourse relation.

        Args:
        a_rel (dict):
          relation in question

        Returns:
        (void)

        """
        rel = EXPLICIT if a_rel[CONNECTIVE].get(TOK_LIST, None) \
            else IMPLICIT
        return rel

    def _normalize_tok_list(self, a_tok_list):
        """Flatten token list, only leaving doc offsets.

        Args:
        a_tok_list (list(list(int)):
          relation in question

        Returns:
        (void)

        """
        return [el[TOK_OFFS_IDX] if isinstance(el, Iterable) else el
                for el in a_tok_list]

    def _dump(self, a_path=None):
        """Dump this model to disc at the given path.

        Args:
        a_path (str or None):
          path to file in which to store the model

        Returns:
        (void)

        """
        if a_path is None:
            return
        # check directory
        dirname = os.path.dirname(a_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        elif not os.path.exists(a_path):
            if not os.access(dirname, os.W_OK) or \
               not os.path.isdir(dirname):
                raise RuntimeError("Cannot write to directory '{:s}'.".format(
                    dirname))
        else:
            if not os.access(a_path, os.W_OK):
                raise RuntimeError("Cannot write to file '{:s}'.".format(
                    a_path))
        with open(a_path, "wb") as ofile:
            dump(self, ofile)
