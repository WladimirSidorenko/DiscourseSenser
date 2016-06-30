#!/usr/bin/env python
# -*- coding: utf-8; mode: python; -*-

"""Module providing class for Wang sense disambiguation.

Attributes:
  WangImplicitSenser (class):
    class that predicts senses of implicit relations using Wang classifier

"""

##################################################################
# Imports
from __future__ import absolute_import, print_function, unicode_literals

from dsenser.constants import ARG1, ARG2, DEPS, \
    DOC_ID, PARSE_TREE, POS, SENTENCES, \
    TOK_LIST
from dsenser.resources import LCSI, BROWN_CLUSTERS, \
    INQUIRER, MPQA, STEMMED_INQUIRER, PSTEMMER, \
    POL_IDX, INTENS_IDX, NEGATIONS
from dsenser.utils import timeit
from dsenser.wang.wangbase import WangBaseSenser


from collections import Counter, defaultdict
from nltk import Tree
from nltk.grammar import is_nonterminal
import re
import sys

##################################################################
# Variables and Constants
DASH_RE = re.compile("-+")
LARROW = "<--"
MODALITY = {"can": 0, "may": 1, "must": 2, "need": 3, "shall": 4,
            "will": 5, "could": 0, "would": 5, "might": 1,
            "should": 4, "'ll": 4, "wo": 5, "sha": 4, "ca": 0,
            "have to": 6, "had to": 6, "'d to": 6, "'ve to": 6
            }
VB_TAG2POS = {"MD": 0, "VB": 1, "VBD": 2, "VBG": 3, "VBN": 4, "VBP": 5,
              "VBZ": 6}
VB_TAGS = set(VB_TAG2POS.iterkeys())
BOTH = "both"
POSITIVE = "positive"


##################################################################
# Classes
class WangImplicitSenser(WangBaseSenser):
    """Class for disambiguating explicit connectives.

    Attrs:
      n_y (int): number of distinct classes

    """

    # private members (used for feature extraction)
    _bc1 = set()
    _bc2 = set()

    def __init__(self, a_clf=None, a_grid_search=False):
        """Class constructor.

        Args:
          a_clf (classifier or None):
            classifier to use or None for default
          a_grid_search (bool): use grid search for estimating hyper-parameters

        """
        super(WangImplicitSenser, self).__init__(a_clf,
                                                 a_grid_search)
        self.n_y = -1
        self.ctype = "implicit"

    @timeit("Training implicit Wang classifier...")
    def train(self, *args, **kwargs):
        super(WangImplicitSenser, self).train(*args, **kwargs)

    def _extract_features(self, a_rel, a_parses):
        """Extract classification features for the given relation.

        Args:
          a_rel (dict):
            discourse relation to extract features for
          a_parses (dict):
            parsed sentences

        Returns:
          void:

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
        self._get_brown_clusters(feats, toks_pos1, toks_pos2)
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
          void:
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
          set:
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
          void:

        Note:
          updates ``a_feats`` dictionary in place

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
          set: set of syntactic productions

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
          a_toks1 (list):
            list of tokens from the 1-st argument
          a_toks2 (list):
            list of tokens from the 2-nd argument

        Returns:
          void:

        Note:
          updates ``a_feats`` dictionary in place

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
          a_toks1 (list):
            list of tokens from the 1-st argument
          a_toks2 (list):
            list of tokens from the 2-nd argument

        Returns:
          void:

        Note:
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
          a_toks (list): argument's tokens

        Returns:
          list:

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
        """Obtain verb classes for the given relation.

        Args:
          a_feats (dict):
            target feature dictionary
          a_toks1 (list(str)):
            list of tokens from the 1-st argument
          a_toks2 (list(str)):
            list of tokens from the 2-nd argument

        Returns:
          void:

        Note:
          updates ``a_feats`` dictionary in place

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
          a_toks (list):
            list of tokens from the 1-st argument

        Returns:
          str: binary mask of activated PoS tags

        """
        ret = [0] * 7
        for _, p in a_toks:
            if p in VB_TAG2POS:
                ret[VB_TAG2POS[p]] = 1
        return ''.join(str(t) for t in ret)

    def _get_brown_clusters(self, a_feats, a_toks1, a_toks2):
        """Obtain Brown cluster pairs for the given relation.

        Args:
          a_feats (dict):
            target feature dictionary
          a_toks1 (list):
            list of tokens from the 1-st argument
          a_toks2 (list):
            list of tokens from the 2-nd argument

        Returns:
          void:

        Note:
          updates ``a_feats`` dictionary in place

        """
        bcluster_str = ""
        self._bc1.update(BROWN_CLUSTERS[w1]
                         for w1, _ in a_toks1 if w1 in BROWN_CLUSTERS)
        self._bc2.update(BROWN_CLUSTERS[w2]
                         for w2, _ in a_toks2 if w2 in BROWN_CLUSTERS)
        # add clusters unique to the first argument
        a_feats.update({"BrownClusterW1-" + bc1: 1
                        for bc1 in self._bc1 - self._bc2})
        # add clusters unique to the second argument
        a_feats.update({"BrownClusterW2-" + bc2: 1
                        for bc2 in self._bc1 - self._bc2})
        # add clusters shared by both arguments
        a_feats.update({"BrownClusterBoth-" + bc: 1
                        for bc in self._bc1 & self._bc2})
        self._bc1.clear()
        self._bc2.clear()

    def _get_inquirer(self, a_feats, a_toks1, a_toks2):
        """Obtain General Inquirer scores for the given relation.

        Args:
          a_feats (dict):
            target feature dictionary
          a_toks1 (list):
            list of tokens from the 1-st argument
          a_toks2 (list):
            list of tokens from the 2-nd argument

        Returns:
          void:

        Note:
          updates ``a_feats`` dictionary in place

        """
        inq1 = self._get_arg_inquirer(a_toks1)
        for i, val in enumerate(inq1):
            if val:
                a_feats["inq1-" + str(i)] = 1.
        inq2 = self._get_arg_inquirer(a_toks2)
        for i, val in enumerate(inq2):
            if val:
                a_feats["inq2-" + str(i)] = 1.
        inq = [i1 and i2 for i1 in inq1 for i2 in inq2]
        for i, val in enumerate(inq):
            if val:
                a_feats["cmnInq-" + str(i)] = 1.

    def _get_arg_inquirer(self, a_toks):
        """Obtain General Inquirer scores for the given relation.

        Args:
          a_toks (list): list of tokens from the first argument

        Returns:
          list:

        """
        ret = [False] * 42
        inq = None
        for w, pos in a_toks:
            if pos not in VB_TAGS:
                continue
            if w in INQUIRER:
                inq = INQUIRER[w]
            else:
                w = PSTEMMER.stem(w)
                if w in INQUIRER:
                    inq = INQUIRER[w]
                elif w in STEMMED_INQUIRER:
                    inq = STEMMED_INQUIRER[w]
            if inq is not None:
                ret = [el1 or el2 for el1, el2 in zip(ret, inq)]
                inq = None
        return ret

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
          void:

        Note:
          updates ``a_feats`` dictionary in place

        """
        mpqa1 = self._get_arg_MPQA(a_toks1)
        mpqa2 = self._get_arg_MPQA(a_toks2)
        a_feats.update(("MPQA1-" + k, v) for k, v in mpqa1.iteritems())
        a_feats.update(("MPQA2-" + k, v) for k, v in mpqa2.iteritems())
        a_feats.update(("MPQA-CMN-" + k, v) for k, v in mpqa1.iteritems()
                       if k in mpqa2)

    def _get_arg_MPQA(self, a_toks):
        """Obtain MPQA scores for single argument.

        Args:
          a_toks (list((str, str))):
            list of argument's tokens and tags

        Returns:
          dict: polarity-intensity values

        """
        ret = {}
        j = 0
        entry = ipol = None
        for i, (tok, pos) in enumerate(a_toks):
            if tok in MPQA:
                entry = MPQA[tok]
                ipol = entry[POL_IDX]
                if ipol == BOTH:
                    continue
                elif ipol == POSITIVE:
                    j = max(0, i - 3)
                    if any(el[0].lower() in NEGATIONS for el in a_toks[j:i]):
                        ipol = "negatedpos"
                ret[ipol + '|' + entry[INTENS_IDX]] = 1.
        return ret
