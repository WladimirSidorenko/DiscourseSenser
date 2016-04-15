#!/usr//bin/env python
# -*- coding: utf-8; mode: python; -*-

"""Module providing class for task-specific word embeddings.

Attributes:
WEMB (class):
  class for fast retrieval and adjustment of the Google word embeddings

"""

##################################################################
# Imports
from __future__ import absolute_import, print_function

from dsenser.resources import W2V

import numpy as np


##################################################################
# Class
class WEMB(object):
    """Class for fast retrieval and adjustment of word embeddings.

    """

    def __init__(self, a_w2v=W2V):
        """Class cosntructor.

        Args:
        a_w2v (gensim.Word2Vec):
        dictionary with the original word embeddings

        """
        self._w2v = a_w2v
        self._cache = {}
        self.ndim = self._w2v.vector_size
        self.default = np.ones(self.ndim)

    def __contains__(self, a_word):
        """Proxy method for looking up a word in the resource.

        Args:
        a_word (str): word to look up in the resource

        Returns:
        (bool):
        true if the word is present in the underlying resource

        """
        if a_word in self._cache:
            return True
        elif a_word in self._w2v:
            self._cache[a_word] = self._w2v[a_word]
            return True
        return False

    def __getitem__(self, a_word):
        """Proxy method for looking up a word in the resource.

        Args:
        a_word (str): word to look up in the resource

        Returns:
        (bool):
        true if the word is present in the underlying resource

        """
        if a_word in self._cache:
            return self._cache[a_word]
        elif a_word in self._w2v:
            emb = self._cache[a_word] = self._w2v[a_word]
            return emb
        raise KeyError
