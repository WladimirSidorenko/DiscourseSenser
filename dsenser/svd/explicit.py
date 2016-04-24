#!/usr/bin/env python
# -*- coding: utf-8; mode: python; -*-

"""Module providing class for SVD sense disambiguation.

Attributes:
SVDExplicitSenser (class):
  class that predicts senses of explicit relations

"""

##################################################################
# Imports
from __future__ import absolute_import, print_function

from dsenser.svd.svdbase import SVDBaseSenser
from dsenser.utils import timeit

##################################################################
# Variables and Constants

##################################################################
# Methods


##################################################################
# Class
class SVDExplicitSenser(SVDBaseSenser):
    """Class for SVD disambiguation of explicit discourse relations.

    Attrs:
    n_y (int): number of distinct classes

    Methods:

    """

    @timeit("Training explicit SVD classifier...")
    def train(self, *args, **kwargs):
        super(SVDExplicitSenser, self).train(*args, **kwargs)
