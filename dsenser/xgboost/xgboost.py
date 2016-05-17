#!/usr/bin/env python
# -*- coding: utf-8; mode: python; -*-

"""Module providing class for XGBoost sense disambiguation.

Attributes:
  XGBoostSenser (class):
    class for XGBoost sense classification of explicit and implicit relations

"""

##################################################################
# Imports
from __future__ import absolute_import, print_function, \
    unicode_literals

from dsenser.wang import WangSenser
from dsenser.xgboost.explicit import XGBoostExplicitSenser
from dsenser.xgboost.implicit import XGBoostImplicitSenser

##################################################################
# Constants


##################################################################
# Class
class XGBoostSenser(WangSenser):
    """Class for XGBoost classification of discourse relations.

    Attributes:
      explicit (:class:`dsenser.xgboost.explicit.XGBoostExplicitSenser`):
        classifier for explicit discourse relations
      implicit (:class:`dsenser.xgboost.implicit.XGBoostImplicitSenser`):
        classifier for implicit discourse relations
      n_y (int): number of distinct classes

    """

    def __init__(self):
        """Class constructor.

        """
        self.explicit = XGBoostExplicitSenser()
        self.implicit = XGBoostImplicitSenser()
        self.n_y = -1
