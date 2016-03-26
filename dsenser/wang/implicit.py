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
from dsenser.constants import ARG1, ARG2, DEPS, DOC_ID, PARSE_TREE, \
    SENTENCES, SENSE, SNT_ID, TOK_ID, TOK_IDX, TOK_LIST, WORDS

from collections import defaultdict
from nltk import Tree
from nltk.grammar import is_nonterminal
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import numpy as np
import re
import sys

##################################################################
# Variables and Constants
DFLT_C = 0.3
DASH_RE = re.compile("-+")
LARROW = "<--"


##################################################################
# Classes
class WangImplicitSenser(BaseSenser):
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
        classifier = LinearSVC(C=DFLT_C, dual=False)
        self._model = Pipeline([('vectorizer', DictVectorizer()),
                                ('var_filter', VarianceThreshold()),
                                ('classifier', classifier)])

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
        print("Training Wang implicit classifier ...", file=sys.stderr)
        self.n_y = a_n_y
        x_train = []
        y_train = []
        x_i = y_i = None
        # generate features
        for irel in a_train_data[0]:
            x_i = self._extract_features(irel, a_train_data[1])
            if not x_i:
                continue
            x_train.append(x_i)
            y_i = np.argmax(irel[SENSE])
            # y_i = [i for i, val in enumerate(irel[SENSE]) if val]
            y_train.append(y_i)
        # fit the model
        # y_train = MultiLabelBinarizer().fit_transform(y_train)
        self._model.fit(x_train, y_train)
        print(" done", file=sys.stderr)

    def predict(self, a_rel, a_data):
        """Method for predicting sense of single relation.

        Args:
        a_rel (dict):
          discourse relation whose sense should be predicted
        a_data (2-tuple(dict, dict)):
          list of input JSON data

        Returns:
        str:
          most probable sense of discourse relation

        """
        ret = np.zeros(self.n_y)
        feats = self._extract_features(a_rel, a_data[-1])
        # map model's classes to original indices
        for i, ival in enumerate(self._model.decision_function(feats)[0]):
            ret[self._model.classes_[i]] = ival
        # normalize using softmax
        return np.exp(ret) / (np.sum(np.exp(ret)) or 1e10)

    def _free(self):
        """Free resources used by the model.

        Args:
        (void):

        Returns:
        (void):

        """
        self.n_y = -1

    def _extract_features(self, a_rel, a_parses):
        """Extract classification features for the given relation.

        Args:
        a_rel (dict):
          discourse relation to extract features for
        a_parses (dict):
          parsed sentences

        Returns:
        (void):

        """
        feats = {}
        doc_id = a_rel[DOC_ID]
        self._get_product_rules(feats, doc_id, a_rel, a_parses)
        self._get_dep_rules(feats, doc_id, a_rel, a_parses)
        self._get_first_last_toks(feats, doc_id, a_rel, a_parses)
        return feats

    def _get_product_rules(self, a_feats, a_doc_id, a_rel, a_parses):
        """Extract syntactic production rules for the given relation.

        Args:
        a_feats (dict):
          target feature dictionary
        a_doc_id (str):
          id of the document
        a_rel (dict):
          discourse relation to extract features for
        a_parses (dict):
          parsed sentences

        Returns:
        (void):
          updates `a_feats` dictionary in place

        """
        arg1_prods = self._get_arg_product_rules(a_doc_id, ARG1,
                                                 a_rel, a_parses)
        arg2_prods = self._get_arg_product_rules(a_doc_id, ARG2,
                                                 a_rel, a_parses)
        both_prods = arg1_prods & arg2_prods
        # update features
        for p in arg1_prods:
            a_feats["Arg1Prod-" + p] = 1.
        for p in arg2_prods:
            a_feats["Arg2Prod-" + p] = 1.
        for p in both_prods:
            a_feats["BothProd-" + p] = 1.

    def _get_arg_product_rules(self, a_doc_id, a_arg, a_rel, a_parses):
        """Extract syntactic production rules for the given arg.

        Args:
        a_doc_id (str):
          id of the document
        a_arg (str):
          argument to extract productions for
        a_rel (dict):
          discourse relation to extract features for
        a_parses (dict):
          parsed sentences

        Returns:
        (set(str)):
          set of syntactic productions

        """
        ret = set()
        # obtain token indices for each arg sentence
        snt_id = None
        snt2tok = self._get_snt2tok(a_rel[a_arg][TOK_LIST])
        # obtain set of leaves corresponding to that argument
        arg_leaves = set()
        subt_leaves = set()
        processed_leaves = set()
        itree = itree_str = inode_path = None
        for snt_id, toks in snt2tok.iteritems():
            itree_str = a_parses[a_doc_id][SENTENCES][snt_id][PARSE_TREE]
            itree = Tree.fromstring(itree_str)
            if not itree.leaves():
                print("Invalid parse tree for sentence {:d}".format(snt_id),
                      file=sys.stderr)
                continue
            # obtain all terminal syntactic nodes from the arg
            for itok in toks:
                inode_path = itree.leaf_treeposition(itok)
                arg_leaves.add(itree[inode_path])
            # check all subtrees (not efficient, but easy to implement)
            for s_t in itree.subtrees():
                subt_leaves.update(s_t.leaves())
                if subt_leaves.issubset(arg_leaves) and \
                   not subt_leaves.issubset(processed_leaves):
                    ret.update(str(p) for p in itree.productions()
                               if any(is_nonterminal(n)
                                      for n in p.rhs()))
                    processed_leaves.update(subt_leaves)
                subt_leaves.clear()
                if processed_leaves == arg_leaves:
                    break
            arg_leaves.clear()
            processed_leaves.clear()
        return ret

    def _get_dep_rules(self, a_feats, a_doc_id, a_rel, a_parses):
        """Extract syntactic dependency rules for the given relation.

        Args:
        a_feats (dict):
          target feature dictionary
        a_doc_id (str):
          id of the document
        a_rel (dict):
          discourse relation to extract features for
        a_parses (dict):
          parsed sentences

        Returns:
        (void):
          updates `a_feats` dictionary in place

        """
        for dep1 in self._get_arg_dep_rules(a_doc_id, ARG1,
                                            a_rel, a_parses):
            a_feats["Dep1-" + dep1] = 1.
        for dep2 in self._get_arg_dep_rules(a_doc_id, ARG2,
                                            a_rel, a_parses):
            a_feats["Dep2-" + dep2] = 1.

    def _get_arg_dep_rules(self, a_doc_id, a_arg, a_rel, a_parses):
        """Extract syntactic dependency rules for the given arg.

        Args:
        a_doc_id (str):
          id of the document
        a_arg (str):
          argument to extract productions for
        a_rel (dict):
          discourse relation to extract features for
        a_parses (dict):
          parsed sentences

        Yields:
        (set(str)):
          set of syntactic productions

        """
        # obtain sentence and token indices
        snt2tok = self._get_snt2tok(a_rel[a_arg][TOK_LIST])
        # obtain dependencies
        word = words = w_id = None
        deps = defaultdict(list)  # prnt: [dep1 dep2]
        for snt_id, itoks in snt2tok.iteritems():
            itoks = set(w_id + 1 for w_id in itoks)
            for idep in a_parses[a_doc_id][SENTENCES][snt_id][DEPS]:
                words = DASH_RE.split(idep[1])
                w_id = int(words.pop()) + 1
                if w_id in itoks:
                    word = '-'.join(words)
                    deps[word].append(idep[0])
        # append dependencies to features
        for head, ideps in deps.iteritems():
            yield head + LARROW + ' '.join(ideps)

    def _get_first_last_toks(self, a_feats, a_doc_id, a_rel, a_parses):
        """Extract first and last tokens of the given relation.

        Args:
        a_feats (dict):
          target feature dictionary
        a_doc_id (str):
          id of the document
        a_rel (dict):
          discourse relation to extract features for
        a_parses (dict):
          parsed sentences

        Returns:
        (void):
          updates `a_feats` dictionary in place

        """
        a1_first3 = a2_first3 = "NULL" * 3
        first1 = last1 = first2 = last2 = first1_first2 = last1_last2 = "NULL"
        # obtain tokens for the first argument
        toks1 = [a_parses[a_doc_id][SENTENCES][s_id][WORDS][w_id][TOK_IDX]
                 for s_id, w_ids in
                 self._get_snt2tok(a_rel[ARG1][TOK_LIST]).iteritems()
                 for w_id in w_ids]
        if toks1:
            first1, last1 = toks1[0], toks1[-1]
            a1_first3 = '_'.join(toks1[:3])
        # obtain tokens for the second argument
        toks2 = [a_parses[a_doc_id][SENTENCES][s_id][WORDS][w_id][TOK_IDX]
                 for s_id, w_ids in
                 self._get_snt2tok(a_rel[ARG2][TOK_LIST]).iteritems()
                 for w_id in w_ids]
        if toks1:
            first2, last2 = toks2[0], toks2[-1]
            a2_first3 = '_'.join(toks2[:3])
        # add obtained tokens to dictionary
        a_feats["Arg1Tok1-" + first1] = 1.
        a_feats["Arg1Tok-1-" + last1] = 1.
        a_feats["Arg1First3-" + a1_first3] = 1.
        a_feats["Arg2Tok1-" + first2] = 1.
        a_feats["Arg2Tok-1-" + last2] = 1.
        a_feats["Arg2First3-" + a2_first3] = 1.
        # add token combinations
        a_feats["Arg1Tok1Arg2Tok1-" + first1 + '|' + first2] = 1.
        a_feats["Arg1Tok1Arg2Tok-1-" + last1 + '|' + last2] = 1.

    def _get_snt2tok(self, a_tok_list):
        """Generate mapping from sentence indices to token lists.

        Args:
        a_tok_list (list(tuple(int, int))):
          list of sentence and token indices pertaining to the argument

        Returns:
        defaultdict(set):
          mapping from sentence indices to token lists

        """
        snt2tok = defaultdict(set)
        for el in a_tok_list:
            snt_id = el[SNT_ID]
            snt2tok[snt_id].add(el[TOK_ID])
        return snt2tok
