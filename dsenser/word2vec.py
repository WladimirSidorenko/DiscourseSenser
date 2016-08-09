#!/usr/bin/env python
# -*- coding: utf-8; mode: python; -*-

"""Module providing class for generic word embeddings.

Attributes:
  WEMB (class):
    class for fast retrieval and adjustment of the Google word embeddings

"""

##################################################################
# Imports
from __future__ import absolute_import, print_function

from dsenser.resources import W2V
from dsenser.utils import singleton

import numpy as np


##################################################################
# Class
@singleton
class Word2Vec(object):
    """Class for cached retrieval of word embeddings.

    """

    def __init__(self, a_w2v=W2V):
        """Class cosntructor.

        Args:
          a_w2v (gensim.Word2Vec):
            dictionary with original word embeddings

        """
        self._w2v = a_w2v
        self._cache = {}
        self.ndim = -1

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

    def load(self):
        """Load the word2vec resource.

        Args:
        (void):

        Returns:
        (void):
        load the resource in place

        """
        self._w2v.load()
        self.ndim = self._w2v.resource.vector_size

    def unload(self):
        """Unload the word2vec resource.

        Args:
        (void):

        Returns:
        (void):
        load the resource in place

        """
        self._cache.clear()
        self._w2v.unload()
