#!/usr/bin/env python
# -*- coding: utf-8; mode: python; -*-

"""Module providing class for SVD sense disambiguation.

Attributes:
  SVDImplicitSenser (class):
    class that predicts senses of implicit relations

"""

##################################################################
# Imports
from __future__ import absolute_import, print_function

from dsenser.svd.svdbase import SVDBaseSenser
from dsenser.utils import timeit

##################################################################
# Variables and Constants


##################################################################
# Classes
class SVDImplicitSenser(SVDBaseSenser):
    """Class for disambiguating implicit discourse relations.

    """

    @timeit("Training implicit SVD classifier...")
    def train(self, *args, **kwargs):
        super(SVDImplicitSenser, self).train(*args, **kwargs)
