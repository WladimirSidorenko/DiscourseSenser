#!/usr/bin/env python
# -*- coding: utf-8; mode: python; -*-

"""Module providing class for Wang sense disambiguation.

Attributes:
WangSenser (class):
  class for predicting sense probabilities for explicit and implcit relations

"""

##################################################################
# Imports
from __future__ import absolute_import, print_function

from dsenser.base import BaseSenser
from dsenser.utils import is_explicit
from dsenser.wang.explicit import WangExplicitSenser
from dsenser.wang.implicit import WangImplicitSenser
from dsenser.wang.wangbase import DFLT_PARAMS, DFLT_EXP_C, DFLT_IMP_C

from sklearn.svm import LinearSVC

##################################################################
# Variables and Constants


##################################################################
# Classes
class WangSenser(BaseSenser):
    """Class using Wang classification for disambiguating connectives.

    Attributes:
      explicit (:class:`dsenser.wang.explicit.WangExplicitSenser`):
        classifier for explicit discourse relations
      implicit (:class:`dsenser.wang.implicit.WangImplicitSenser`):
        classifier for implicit discourse relations
      n_y (int): number of distinct classes

    """

    def __init__(self, **kwargs):
        """Class constructor.

        Args:
          kwargs (dict): keyword arguments to be forwarded

        """
        explicit_clf = LinearSVC(C=DFLT_EXP_C, **DFLT_PARAMS)
        self.explicit = WangExplicitSenser(a_clf=explicit_clf, **kwargs)
        implicit_clf = LinearSVC(C=DFLT_IMP_C, **DFLT_PARAMS)
        self.implicit = WangImplicitSenser(a_clf=implicit_clf, **kwargs)
