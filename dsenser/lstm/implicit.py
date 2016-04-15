#!/usr//bin/env python
# -*- coding: utf-8; mode: python; -*-

"""Module providing class for LSTM sense disambiguation.

Attributes:
LSTMImplicitSenser (class):
  class that predicts senses of implicit relations

"""

##################################################################
# Imports
from __future__ import absolute_import, print_function

from dsenser.lstm.base import LSTMBaseSenser
from dsenser.utils import timeit
from dsenser.theano_utils import floatX, rmsprop

import sys

##################################################################
# Variables and Constants


##################################################################
# Classes
class LSTMImplicitSenser(LSTMBaseSenser):
    """Class for disambiguating explicit connectives.

    Attrs:
    n_y (int): number of distinct classes

    Methods:

    """

    @timeit("Training implicit LSTM classifier...")
    def train(self, *args, **kwargs):
        super(LSTMImplicitSenser, self).train(*args, **kwargs)
