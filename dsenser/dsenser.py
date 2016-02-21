#!/usr//bin/env python
# -*- coding: utf-8; mode: python; -*-

"""Module providing class for sense disambiguation of connectives.

Attributes:
DiscourseSenser (class): class for sense disambiguation of connectives

"""

##################################################################
# Imports
from __future__ import absolute_import, print_function, unicode_literals

from cPickle import dump, load

from dsenser.constants import DFLT_MODEL, FFNN

##################################################################
# Variables and Constants


##################################################################
# Class
class DiscourseSenser(object):
    """Class for sense disambiguation of connectives.

    Attributes:

    """

    def __init__(self, a_model=None):
        """Class constructor.

        Args:
        a_model (str or None): path to serialized model or None

        """
        self.model = None
        # load serialized model
        if a_model is not None:
            self.model = load(a_model)

    def train(self, a_train_data, a_model=DFLT_MODEL, a_type=FFNN,
              a_dev_data=None):
        """Train specified model on the provided data.

        Args:
        a_dev_data (list or None):
          development set
        a_model (str):
          path for storing the model
        a_type (str):
          type of the model to be trained
        a_dev_data (list or None):
          development set

        Returns:
          (void)

        """
        pass

    def predict(self, a_data):
        """Determine senses of discourse connectives.

        Args:
        a_data (list):
          input data to be analyzed

        Returns:
        (void):
          updates input set in place

        """
        if self.model is None:
            raise RuntimeError(
                "No trained model is provided to make predictions.")
        raise NotImplementedError
