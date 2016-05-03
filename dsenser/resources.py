#!/usr//bin/env python
# -*- coding: utf-8; mode: python; -*-

"""Module providing access to third party resources.

Attributes:
load_conns (lambda): method for loading explicit connectives
conns2str (lambda): connective normalization method
load_LCSI (lambda): method for loading LCSI verb classes
load_BROWN (lambda): method for loading Brown clusters
LCSI (dict(str: set(str))):  mapping from verb to a set of classes
BROWN_CLUSTERS (dict(str: str)): mapping from word to a set of
            Brown clusters
CONNS (set(str)): set of explcit connectives
CONNTOK2CONN (defaultdict(list)): mapping from connective to its
            enumerated tokens
CONNTOKS (set): set of tokens which can be part of a connective
INQUIRER (dict): mapping from word to General Inquirer class
STEMMED_INQUIRER (dict): mapping from stemmed word to General Inquirer class
W2V (dict): word2vec embeddings

"""

##################################################################
# Imports
from __future__ import absolute_import, print_function

from dsenser.constants import ENCODING, DFLT_BROWN_PATH, DFLT_ECONN_PATH, \
    DFLT_INQUIRER_PATH, DFLT_LCSI_PATH, DFLT_MPQA_PATH, DFLT_W2V_PATH
from dsenser.scorer.conn_head_mapper import ConnHeadMapper

from collections import defaultdict
from nltk.stem.porter import PorterStemmer
import codecs
import gc
import re
import sys


##################################################################
# Constants
BAR_RE = re.compile(r'\|')
CHM = ConnHeadMapper()
ELLIPSIS_RE = re.compile(r"[.][.]+")
EQ_RE = re.compile("=+")
HASH_RE = re.compile("\s*#\s*")
SPACE_RE = re.compile("\s+")
TAB_RE = re.compile("\t+")
PSTEMMER = PorterStemmer()
WORD1 = "word1"
POL = "priorpolarity"
POL_IDX = 0
INTENS = "type"
INTENS_IDX = 1
POS = "pos1"
POS_IDX = 2
NEGATIONS = set(["cannot", "not", "none", "nothing",
                 "nowhere", "neither", "nor", "nobody",
                 "hardly", "scarcely", "barely", "never",
                 "n't", "noone", "havent", "hasnt",
                 "hadnt", "cant", "couldnt", "shouldnt",
                 "wont", "wouldnt", "dont", "doesnt",
                 "didnt", "isnt", "arent", "aint", "no"
                 ])


##################################################################
# Methods
def load_conns(a_fname):
    """Load explicit connectives from file.

    Args:
    a_fname (str): file containing connectives

    Returns:
    (set(str)): set of loaded connectives

    """
    ret = set()
    iconn = None
    with codecs.open(a_fname, 'r', ENCODING) as ifile:
        for iline in ifile:
            iline = iline.strip().lower()
            if not iline:
                continue
            iconn = tuple(tuple(itok.split()) for itok
                          in ELLIPSIS_RE.split(iline))
            ret.add(iconn,)
    return ret


def conn2str(a_conn):
    """Convert connective tuple to string.

    Args:
    a_conn (tuple):
    tuple of connective tokens

    Returns:
    (str): connective string

    """
    return '_'.join(itok for ipart in a_conn for itok in ipart)


def load_LCSI(a_fname):
    """Load LCSI verb classes from file.

    Args:
    a_fname (str): file containing LCSI data

    Returns:
    (dict(str: set(str))):
    mapping from verb to a set of classes

    """
    ret = dict()
    iword = iclasses = iclass_str = None
    with codecs.open(a_fname, 'r', ENCODING) as ifile:
        for iline in ifile:
            iline = iline.strip()
            if not iline:
                continue
            iword, iclass_str = SPACE_RE.split(iline, 1)
            iword = iword.lower()
            iclasses = set(HASH_RE.split(iclass_str))
            ret[iword] = iclasses
    return ret


def load_BROWN(a_fname):
    """Load BROWN clusters from file.

    Args:
    a_fname (str): file containing Brown clusters

    Returns:
    (dict(str: set(str))):
    mapping from word to a set of Brown clusters

    """
    ret = defaultdict(set)
    iword = iclass = None
    print("Loading {:s}... ".format(a_fname), end="", file=sys.stderr)
    with codecs.open(a_fname, 'r', ENCODING) as ifile:
        for iline in ifile:
            iline = iline.strip()
            if not iline:
                continue
            _, iword, iclass = SPACE_RE.split(iline, 2)
            iword = iword.lower()
            ret[iword].add(iclass)
    print("done", file=sys.stderr)
    # convert defaultdict back to the normal one
    return {k: '|'.join(cls) for k, cls in ret.iteritems()}


def load_INQUIRER(a_fname):
    """Load Inquirer data from file.

    Args:
    a_fname (str): file containing Inquirer data

    Returns:
    (dict(str: []), dict(str: [])):
      mapping from word to Inquirer classes and the same mapping for stemmed
      words

    """
    ret = dict()
    stem_ret = dict()
    iword = iclass = None
    with codecs.open(a_fname, 'r', ENCODING) as ifile:
        for iline in ifile:
            iline = iline.strip()
            if not iline:
                continue
            iword, iclass = TAB_RE.split(iline, 1)
            iword = iword.strip().lower()
            iclass = [bool(el) if el else False for el in BAR_RE.split(iclass)]
            ret[iword] = iclass
            ret[PSTEMMER.stem(iword)] = iclass
    # convert defaultdict back to the normal one
    return (ret, stem_ret)


def load_MPQA(a_fname):
    """Load MPQA data from file.

    Args:
    a_fname (str): file containing MPQA data

    Returns:
    (dict(str: tuple(polarity, type, pos))):
      mapping from word to Inquirer classes and the same mapping for stemmed
      words

    """
    ret = defaultdict(lambda: [None] * 3)
    attrs = None
    with codecs.open(a_fname, 'r', ENCODING) as ifile:
        for iline in ifile:
            iline = iline.strip()
            if not iline:
                continue
            attrs = dict(EQ_RE.split(iword.lower(), 1)
                         for iword in SPACE_RE.split(iline)
                         if EQ_RE.search(iword))
            ret[attrs[WORD1]] = (attrs[POL], attrs[INTENS], attrs[POS])
    # convert defaultdict back to the normal one
    return dict(ret.iteritems())


def load_W2V(a_fname):
    """Load Word2Vec data from file.

    Args:
    a_fname (str): file containing W2V data

    Returns:
    (dict(str: tuple(polarity, type, pos))):
      mapping from word to Inquirer classes and the same mapping for stemmed
      words

    """
    from gensim.models.word2vec import Word2Vec
    print("Loading {:s}... ".format(a_fname), end="", file=sys.stderr)
    w2v = Word2Vec.load_word2vec_format(a_fname, binary=True)
    print("done", file=sys.stderr)
    return w2v


##################################################################
# Class
class LoadOnDemand(object):
    """Custom class for deferring loading of huge resources.

    Loads resources only if they are actually used.

    Attributes:
    __init__ (method): class constructor
    __getattr__ (method): method for initiliazing the resource the first time
                       it gets used
    resource (object or None): loaded resource
    cmd (method): method to load the  resource
    args (list): arguments to pass to ``cmd``
    kwargs (dict): keyword arguments to pass to ``cmd``

    """

    def __init__(self, a_cmd, *a_args, **a_kwargs):
        """Class cosntructor.

        Args:
        a_cmd (method): custom method to load the resource
        args (list): arguments to pass to ``a_cmd``
        kwargs (dict): keyword arguments to pass to ``a_cmd``

        """
        self.resource = None
        self.cmd = a_cmd
        self.args = a_args
        self.kwargs = a_kwargs

    def __contains__(self, a_name):
        """Proxy method for looking up a word in the resource.

        Args:
        a_name (str): word to look up in the resource

        Returns:
        forwards the request to the underlying resource

        """
        self.load()
        return a_name in self.resource

    def __getitem__(self, a_name):
        """Proxy method for accessing the resource.

        Args:
        a_name (str): word to look up in the resource

        Returns:
        forwards the request to the underlying resource

        """
        # initialize the resource if needed
        self.load()
        return self.resource.__getitem__(a_name)

    def load(self):
        """Force loading the resource.

        Args:
        (void):

        Returns:
        (void):
        load the resource

        """
        if self.resource is None:
            self.resource = self.cmd(*self.args, **self.kwargs)
        return self.resource

    def unload(self):
        """Unload the resource.

        Args:
        (void):

        Returns:
        (void):
        load the resource

        """
        if self.resource is not None:
            print("Unloading resource '{:s}'...".format(repr(self.resource)),
                  file=sys.stderr)
            del self.resource
            self.resource = None
            gc.collect()


##################################################################
# Resources
LCSI = load_LCSI(DFLT_LCSI_PATH)
BROWN_CLUSTERS = LoadOnDemand(load_BROWN, DFLT_BROWN_PATH)
CONNS = load_conns(DFLT_ECONN_PATH)
CONNTOK2CONN = defaultdict(list)

itok = None
for iconn in CONNS:
    for i, ipart in enumerate(iconn):
        itok = ipart[0]
        CONNTOK2CONN[itok].append((i, iconn))

for iconns in CONNTOK2CONN.itervalues():
    iconns.sort(key=lambda el: el[0])

CONNTOKS = set(CONNTOK2CONN.keys())
INQUIRER, STEMMED_INQUIRER = load_INQUIRER(DFLT_INQUIRER_PATH)
MPQA = load_MPQA(DFLT_MPQA_PATH)
W2V = LoadOnDemand(load_W2V, DFLT_W2V_PATH)
