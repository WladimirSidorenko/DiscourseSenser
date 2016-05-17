#!/usr/bin/env python
# -*- coding: utf-8; mode: python; -*-

"""Module providing class for LSTM sense disambiguation.

Attributes:
  LSTMSenser (class):
    class for predicting sense probabilities for explicit and implcit relations

"""

##################################################################
# Imports
from __future__ import absolute_import, print_function

from dsenser.base import BaseSenser
from dsenser.lstm.explicit import LSTMExplicitSenser
from dsenser.lstm.implicit import LSTMImplicitSenser


##################################################################
# Class
class LSTMSenser(BaseSenser):
    """Class using LSTM classification for disambiguating connectives.

    Attributes:
      explicit (:class:`dsenser.lstm.explicit.LSTMExplicitSenser`):
        classifier for implicit discourse relations
      implicit (:class:`dsenser.lstm.implicit.LSTMImplicitSenser`):
        classifier for explicit discourse relations
      n_y (int): number of distinct classes

    """

    def __init__(self, *args, **kwargs):
        """Class constructor.

        Args:
          args (list): list of arguments
          kwargs (dict): dictionary of keyword arguments

        """
        self.explicit = LSTMExplicitSenser(*args, **kwargs)
        self.implicit = LSTMImplicitSenser(*args, **kwargs)
        self.n_y = -1
