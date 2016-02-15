#!/usr//bin/env python
# -*- coding: utf-8; mode: python; -*-

"""Module providing class for shallow discourse parsing.

Attributes:
DiscourseParser (class): main discourse parser class

"""

##################################################################
# Imports
from __future__ import absolute_import

from dsegmenter.constants import DFLT_SEGM_MODEL, DFLT_SENSE_MODEL

from cPickle import dump, load


##################################################################
# Class
class DiscourseParser(object):
    """Main class for doing shallow discourse parsing.

    Attributes:
    segm_model (object): model for doing discourse segmentation
    sense_model (object): model for doing sense disambiguation

    """

    def __init__(self, a_segm_model=DFLT_SEGM_MODEL,
                 a_sense_model=DFLT_SENSE_MODEL):
        """Class constructor.

        Args:
        a_segm_model: None or str
          path to serialized sement model
        a_sense_model: None or str
          path to serialized sense disambiguation model

        """
        self.segm_model = a_segm_model if a_segm_model is None \
            else load(a_segm_model)
        self.sense_model = a_sense_model if a_sense_model is None \
            else load(a_sense_model)

    def train(self, a_segm_model=None, a_sense_model=None):
        """Class constructor.

        Args:
        a_segm_model: None or str
          path to serialized sement model
        a_sense_model: None or str
          path to serialized sense disambiguation model

        Returns:
        (void)

        """
        assert a_segm_model or a_sense_model, \
            "At least one of trained components should be not-None"
        if a_segm_model is not None:
            raise NotImplementedError
            self.train_segmenter(a_segm_model)
        if a_sense_model is not None:
            self.train_sense(a_sense_model)

    def train_senser(self, a_sense_model):
        """Class constructor.

        Args:
        a_sense_model: str
          path to serialized sense disambiguation model

        Returns:
        (void)

        """
