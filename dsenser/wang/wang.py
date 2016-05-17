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

    def __init__(self):
        """Class constructor.

        """
        self.explicit = WangExplicitSenser()
        self.implicit = WangImplicitSenser()
        self.n_y = -1
