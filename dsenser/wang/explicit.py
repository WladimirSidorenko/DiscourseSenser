#!/usr//bin/env python
# -*- coding: utf-8; mode: python; -*-

"""Module providing class for Wang sense disambiguation.

Attributes:
WangSenser (class):
  class that always chooses majority category  for sense disambiguation

"""

##################################################################
# Imports
from __future__ import absolute_import, print_function

from dsenser.base import BaseSenser
from dsenser.constants import CONNECTIVE, DOC_ID, TOK_LIST, \
    SENTENCES, TOK_IDX

import numpy as np

##################################################################
# Variables and Constants
ENCODING = "utf-8"
PREV_NONE = "prev1_NONE"

##################################################################
# Methods


##################################################################
# Classes
class WangExplicitSenser(BaseSenser):
    """Class for disambiguating explicit connectives.

    Attrs:
    n_y (int): number of distinct classes

    Methods:

    """

    def __init__(self):
        """Class constructor.

        Args:

        """
        self.n_y = -1

    def train(self, a_train_data, a_dev_data=None, a_n_y=-1):
        """Method for training the model.

        Args:
        a_train_data (2-tuple(list, dict)):
          list of training JSON data
        a_dev_data (2-tuple(list, dict) or None):
          list of development JSON data
        a_n_y (int):
          number of distinct classes

        Returns:
        (void)

        """
        self.n_y = a_n_y

    def predict(self, a_rel, a_test_data):
        """Method for predicting sense of single relation.

        Args:
        a_rel (dict):
          discourse relation whose sense should be predicted
        a_test_data (2-tuple(dict, dict)):
          list of input JSON data

        Returns:
        str:
          most probable sense of discourse relation

        """
        pass

    def _free(self):
        """Free resources used by the model.

        Args:
        (void):

        Returns:
        (void):

        """
        self.n_y = -1

    def _extract_features(self, a_rel, a_parses):
        """Free resources used by the model.

        Args:
        a_rel (dict):
          discourse relation to extract features for
        a_parses (dict):
          parsed data

        Returns:
        (void):

        """
        feats = []
        doc_id = a_rel[DOC_ID]
        # obtain raw connective string
        conn_str = self._get_conn_txt(doc_id, a_rel, a_parses)
        feats.append(conn_str)
        feats.append(conn_str.lower())
        # obtain token preceding the connective
        prev_tok = self._get_conn_prev(doc_id, a_rel, a_parses)
        feats.append("{:s}|{:s}".format(conn_str, prev_tok))
        # pitler's syntactic features
        prnt_cat = None
        lft_sib_cat = None
        rght_sib_cat = None
        return feats

    def _get_conn_text(self, a_doc_id, a_rel, a_parses):
        """Obtain raw text of the connective.

        Args:
        a_doc_id (str):
          id of the document containing the connectiv
        a_rel (dict):
          discourse relation to extract features for
        a_parses (dict):
          parsed data

        Returns:
        (str): connective's text

        """
        return ' '.join([
            a_parses[a_doc_id][SENTENCES][snt_id][tok_id][TOK_IDX]
            for _, _, _, snt_id, tok_id in a_rel[CONNECTIVE][TOK_LIST]])

    def _get_conn_prev(self, doc_id, a_rel, a_parses):
        """Obtain raw text of the connective.

        Args:
        a_doc_id (str):
          id of the document containing the connectiv
        a_rel (dict):
          discourse relation to extract features for
        a_parses (dict):
          parsed data

        Returns:
        (str): connective's text

        """
        _, _, _, snt_id, tok_id = a_rel[CONNECTIVE][TOK_LIST][0]
        if tok_id > 0:
            tok_id -= 1
        elif snt_id > 0:
            snt_id == 1
            tok_id = 1
        else:
            return PREV_NONE
        return a_parses[a_doc_id][SENTENCES][snt_id][tok_id][TOK_IDX]
