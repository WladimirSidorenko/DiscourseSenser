#!/usr/bin/env python
# -*- coding: utf-8; mode: python; -*-

"""Module providing class for SVD sense disambiguation.

"""

##################################################################
# Imports
from __future__ import absolute_import, print_function

from dsenser.base import BaseSenser
from dsenser.svd.explicit import SVDExplicitSenser
from dsenser.svd.implicit import SVDImplicitSenser


##################################################################
# Class
class SVDSenser(BaseSenser):
    """Class using LSTM classification for disambiguating connectives.

    Attributes:
      explicit (:class:`dsenser.svd.explicit.SVDExplicitSenser`):
        classifier for implicit discourse relations
      implicit (:class:`dsenser.svd.implicit.SVDExplicitSenser`):
        classifier for explicit discourse relations
      n_y (int): number of distinct classes

    """

    def __init__(self, *args, **kwargs):
        """
        Class constructor.

        Args:
          args (list): list of arguments
          kwargs (dict): dictionary of keyword arguments

        """
        self.explicit = SVDExplicitSenser(*args, **kwargs)
        self.implicit = SVDImplicitSenser(*args, **kwargs)
        self.n_y = -1
