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
from dsenser.constants import ARG1, ARG2, DFLT_LCSI_PATH, DEPS, \
    DOC_ID, ENCODING, PARSE_TREE, POS, SENTENCES, SENSE, SNT_ID, \
    TOK_ID, TOK_IDX, TOK_LIST, WORDS

from collections import Counter, defaultdict
from nltk import Tree
from nltk.grammar import is_nonterminal
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import codecs
import numpy as np
import re
import sys

##################################################################
# Variables and Constants
DFLT_C = 0.3
DASH_RE = re.compile("-+")
SPACE_RE = re.compile("\s+")
HASH_RE = re.compile("\s*#\s*")
LARROW = "<--"
MODALITY = {"can": 0, "may": 1, "must": 2, "need": 3, "shall": 4,
            "will": 5, "could": 0, "would": 5, "might": 1,
            "should": 4, "'ll": 4, "wo": 5, "sha": 4, "ca": 0,
            "have to": 6, "had to": 6, "'d to": 6, "'ve to": 6
            }
VB_TAG2POS = {"MD": 0, "VB": 1, "VBD": 2, "VBG": 3, "VBN": 4, "VBP": 5,
              "VBZ": 6}
VB_TAGS = set(["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"])


##################################################################
# Methods
def load_LCSI(a_fname):
    """Load LCSI verb classes from file.

    Args:
    a_fname (str): file containing connectives

    Returns:
    (dict(str: ste(str))):
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


LCSI = load_LCSI(DFLT_LCSI_PATH)


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
        toks_pos1 = self._get_toks_pos(a_parses[doc_id][SENTENCES],
                                       a_rel, ARG1)
        toks_pos2 = self._get_toks_pos(a_parses[doc_id][SENTENCES],
                                       a_rel, ARG2)

        self._get_product_rules(feats, doc_id, a_rel, a_parses)
        self._get_dep_rules(feats, doc_id, a_rel, a_parses)
        self._get_first_last_toks(feats, toks_pos1, toks_pos2)
        self._get_modality(feats, toks_pos1, toks_pos2)
        self._get_vb_class(feats, toks_pos1, toks_pos2)
        self._get_bclusters(feats, toks_pos1, toks_pos2)
        self._get_inquirer(feats, toks_pos1, toks_pos2)
        self._get_MPQA(feats, toks_pos1, toks_pos2)
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

    def _get_first_last_toks(self, a_feats, a_toks1, a_toks2):
        """Extract first and last tokens of the given relation.

        Args:
        a_feats (dict):
          target feature dictionary
        a_toks1 (list(str)):
          list of tokens from the 1-st argument
        a_toks2 (list(str)):
          list of tokens from the 2-nd argument

        Returns:
        (void):
          updates `a_feats` dictionary in place

        """
        a1_first3 = a2_first3 = "NULL" * 3
        first1 = last1 = first2 = last2 = "NULL"
        # obtain tokens for the first argument
        if a_toks1:
            first1, last1 = a_toks1[0][0], a_toks1[-1][0]
            a1_first3 = '_'.join(t[0] for t in a_toks1[:3])
        # obtain tokens for the second argument
        if a_toks2:
            first2, last2 = a_toks2[0][0], a_toks2[-1][0]
            a2_first3 = '_'.join(t[0] for t in a_toks2[:3])
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

    def _get_modality(self, a_feats, a_toks1, a_toks2):
        """Estimate modality of the given relation.

        Args:
        a_feats (dict):
          target feature dictionary
        a_toks1 (list(str)):
          list of tokens from the 1-st argument
        a_toks2 (list(str)):
          list of tokens from the 2-nd argument

        Returns:
        (void):
          updates `a_feats` dictionary in place

        """
        mod1 = self._get_arg_modality(a_toks1)
        mod2 = self._get_arg_modality(a_toks2)
        joint_mod = [i * j for i in mod1 for j in mod2]
        # add modality features
        a_feats["Mod1-" + ''.join(str(i) for i in mod1)] = 1.
        a_feats["Mod2-" + ''.join(str(i) for i in mod2)] = 1.
        a_feats["JointMod-" + ''.join(str(i) for i in joint_mod)] = 1.

    def _get_arg_modality(self, a_toks):
        """Estimate modality of the given relation argument.

        Args:
        a_toks (list(str)):
          argument's tokens

        Returns:
        (list(int)):

        """
        ret = [0] * 7
        bigram = None
        max_i = len(a_toks) - 1
        for i, itok in enumerate(a_toks):
            itok = itok[0]
            if itok in MODALITY:
                ret[MODALITY[itok]] = 1.
            if i < max_i:
                bigram = itok + ' ' + a_toks[i + 1][0]
                if bigram in MODALITY:
                    ret[MODALITY[bigram]] = 1.
        return ret

    def _get_vb_class(self, a_feats, a_toks1, a_toks2):
        """Obtain verb classes of the given relation.

        Args:
        a_feats (dict):
          target feature dictionary
        a_toks1 (list(str)):
          list of tokens from the 1-st argument
        a_toks2 (list(str)):
          list of tokens from the 2-nd argument

        Returns:
        (void):
          updates `a_feats` dictionary in place

        """
        # find intersecting verb classes
        vb_classes = Counter()
        vb_cls1 = vb_cls2 = None
        for w1, p1 in a_toks1:
            if w1 not in LCSI or p1 not in VB_TAGS:
                continue
            vb_cls1 = LCSI[w1]
            for w2, p2 in a_toks2:
                if w2 not in LCSI or p2 not in VB_TAGS:
                    continue
                vb_cls2 = LCSI[w2]
                vb_classes.update(vb_cls1 & vb_cls2)
        for vbc, cnt in vb_classes.iteritems():
            a_feats["LCSI-" + vbc] = cnt
        # obtain VB tag vectors
        a_feats["VBTags1-" + self._get_arg_vb_class(a_toks1)] = 1.
        a_feats["VBTags2-" + self._get_arg_vb_class(a_toks2)] = 1.

    def _get_arg_vb_class(self, a_toks):
        """Obtain verb classes of single argument.

        Args:
        a_toks (list(str)):
          list of tokens from the 1-st argument

        Returns:
        (str):
          binary mask of activated PoS tags

        """
        ret = [0] * 7
        for _, p in a_toks:
            if p in VB_TAG2POS:
                ret[VB_TAG2POS[p]] = 1.
        return ''.join(str(t) for t in ret)

    def _get_bclusters(self, a_feats, a_toks1, a_toks2):
        """Obtain Brown cluster pairs for the given relation.

        Args:
        a_feats (dict):
          target feature dictionary
        a_toks1 (list(str)):
          list of tokens from the 1-st argument
        a_toks2 (list(str)):
          list of tokens from the 2-nd argument

        Returns:
        (void):
          updates `a_feats` dictionary in place

        """
        pass

    def _get_inquirer(self, a_feats, a_toks1, a_toks2):
        """Obtain General Inquirer scores for the given relation.

        Args:
        a_feats (dict):
          target feature dictionary
        a_toks1 (list(str)):
          list of tokens from the 1-st argument
        a_toks2 (list(str)):
          list of tokens from the 2-nd argument

        Returns:
        (void):
          updates `a_feats` dictionary in place

        """
        pass

    def _get_MPQA(self, a_feats, a_toks1, a_toks2):
        """Obtain MPQA scores for the given relation.

        Args:
        a_feats (dict):
          target feature dictionary
        a_toks1 (list(str)):
          list of tokens from the 1-st argument
        a_toks2 (list(str)):
          list of tokens from the 2-nd argument

        Returns:
        (void):
          updates `a_feats` dictionary in place

        """
        pass

    def _get_snt2tok(self, a_tok_list):
        """Generate mapping from sentence indices to token lists.

        Args:
        a_tok_list (list(tuple(int, int))):
          list of sentence and token indices pertaining to the argument

        Returns:
        defaultdict(set):
          mapping from sentence indices to token lists

        """
        snt2tok_pos = defaultdict(set)
        for el in a_tok_list:
            snt_id = el[SNT_ID]
            snt2tok_pos[snt_id].add(el[TOK_ID])
        return snt2tok_pos

    def _get_toks_pos(self, a_parses, a_rel, a_arg):
        """Method for getting raw tokens with their parts of speech.

        Args:
        a_parses (dict):
          parsed sentences
        a_rel (dict):
          discourse relation whose tokens should be obtained
        a_arg (str):
          relation argument to obtain senses for

        Returns:
        (list(tuple(str, str))):
          list of tokens and their parts of speech

        """
        ret = []
        snt = wrd = None
        for s_id, w_ids in \
                self._get_snt2tok(a_rel[a_arg][TOK_LIST]).iteritems():
            snt = a_parses[s_id][WORDS]
            for w_id in w_ids:
                wrd = snt[w_id]
                ret.append((wrd[TOK_IDX].lower(), wrd[1][POS]))
        return ret
