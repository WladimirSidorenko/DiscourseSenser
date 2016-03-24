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
from dsenser.constants import CONNECTIVE, DFLT_ECONN_PATH, DOC_ID, \
    TOK_LIST, SENTENCES, WORDS, POS, TOK_IDX, SNT_ID, TOK_ID, PARSE_TREE

from collections import defaultdict
from nltk import Tree
import codecs
import numpy as np
import re
import sys

##################################################################
# Variables and Constants
ENCODING = "utf-8"
PREV_NONE = "prev1_NONE"
SPACE_RE = re.compile(r"\s+")
ELLIPSIS_RE = re.compile(r"[.][.]+")
LEFT = 1
RIGHT = 2
DFLT_PRNT = "SBAR"
AS = "as"
WHEN = "when"
CONNS = None
CONNTOKS = None
CONNTOK2CONN = defaultdict(list)


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


def _conn2str(a_conn):
    """Convert connective tuple to string.

    Args:
    a_conn (tuple):
    tuple of connective tokens

    Returns:
    (str): connective string

    """
    pass


##################################################################
# Load resources
CONNS = load_conns(DFLT_ECONN_PATH)

itok = None
for iconn in CONNS:
    for i, ipart in enumerate(iconn):
        itok = ipart[0]
        CONNTOK2CONN[itok].append((i, iconn))

for iconns in CONNTOK2CONN.itervalues():
    iconns.sort(key=lambda el: el[0])

CONNTOKS = set(CONNTOK2CONN.keys())


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
        x_train = [self._extract_features(irel, a_train_data[1])
                   for irel in a_train_data[0]]
        print("a_train_data =", repr(a_train_data), file=sys.stderr)

    def predict(self, a_rel, a_test_data):
        """Method for predicting sense of single relation.

        Args:
        a_rel (dict):
          discourse relation whose sense should be predicted
        a_test_data (2-tuple(dict, dict)):
          list of input JSON data

        Returns:
        (str):
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
        snt_id = a_rel[CONNECTIVE][TOK_LIST][-1][SNT_ID]

        ###########################
        # connective token features
        conn_tok = self._get_conn_txt(doc_id, a_rel, a_parses)
        conn_ltok = conn_tok.lower()
        assert conn_tok, \
            "Connective string cannot be empty for explicit classifier."
        conn_str = "ConStr-" + conn_tok
        feats.append(conn_str)
        conn_lstr = conn_str.lower()
        feats.append(conn_lstr)
        # obtain POS of the connective
        conn_pos = "conpos-" + self._get_conn_pos(doc_id, a_rel, a_parses)
        feats.append(conn_pos)
        # obtain token preceding the connective
        prev_tok = "prevtok-" + self._get_conn_prev(doc_id, a_rel, a_parses)
        feats.append("Tok1Tok2-{:s}|{:s}".format(conn_str, prev_tok))
        ####################
        # syntactic features
        tree_str = a_parses[doc_id][SENTENCES][snt_id][PARSE_TREE].strip()
        parse_tree = Tree.fromstring(tree_str)
        conn_t_ids = [t[-1] for t in a_rel[CONNECTIVE][TOK_LIST]]
        scat = "SyntCat-" + self._get_cat(conn_t_ids, parse_tree)
        feats.append(scat)
        prnt_cat = "PrntCat-" + self._get_prnt_cat(conn_t_ids, parse_tree)
        feats.append(prnt_cat)
        left_sib = "LeftSib-" + self._get_sib(conn_t_ids, parse_tree, LEFT)
        feats.append(left_sib)
        right_sib = "RightSib-" + self._get_sib(conn_t_ids, parse_tree, RIGHT)
        feats.append(right_sib)

        ################
        # joint features
        feats.append(conn_lstr + '|' + scat)
        feats.append(conn_lstr + '|' + prnt_cat)
        feats.append(conn_lstr + '|' + left_sib)
        feats.append(conn_lstr + '|' + right_sib)

        feats.append(scat + '|' + prnt_cat)
        feats.append(scat + '|' + left_sib)
        feats.append(scat + '|' + right_sib)
        feats.append(prnt_cat + '|' + left_sib)
        feats.append(prnt_cat + '|' + right_sib)
        feats.append(left_sib + '|' + right_sib)
        #################
        # Wang's features
        ctx_nodes = "CtxNodes-" + '-'.join(self._get_ctx_nodes(conn_t_ids,
                                                               parse_tree))

        if conn_ltok == AS or conn_ltok == WHEN:
            prev_conn = self._get_prev_conn(conn_t_ids[0], parse_tree)
            if prev_conn:
                feats.append(conn_ltok + '-' + _conn2str(prev_conn))
            print("prev_conn =", repr(prev_conn))
            print("feats =", repr(feats))
            # raise NotImplementedError
            # sys.exit(66)
        if conn_ltok != AS:
            feats.append("NotAs")
        if conn_ltok != WHEN:
            feats.append("NotWhen")

        ###################
        # Digitize features
        # pitler's syntactic features
        prnt_cat = None
        lft_sib_cat = None
        rght_sib_cat = None
        return feats

    def _get_conn_txt(self, a_doc_id, a_rel, a_parses):
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
            a_parses[a_doc_id][SENTENCES][snt_id][WORDS][tok_id][TOK_IDX]
            for _, _, _, snt_id, tok_id in a_rel[CONNECTIVE][TOK_LIST]])

    def _get_conn_pos(self, a_doc_id, a_rel, a_parses):
        """Obtain part-of-speech tags of the connective.

        Args:
        a_doc_id (str):
          id of the document containing the connectiv
        a_rel (dict):
          discourse relation to extract features for
        a_parses (dict):
          parsed data

        Returns:
        (str):
        connective's part-of-speech tags

        """
        return '_'.join([
            a_parses[a_doc_id][SENTENCES][snt_id][WORDS][tok_id][1][POS]
            for _, _, _, snt_id, tok_id in a_rel[CONNECTIVE][TOK_LIST]])

    def _get_conn_prev(self, a_doc_id, a_rel, a_parses):
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
            snt_id -= 1
            tok_id = -1
        else:
            return PREV_NONE
        return a_parses[a_doc_id][SENTENCES][snt_id][WORDS][tok_id][TOK_IDX]

    def _get_cat(self, a_conn_toks, a_tree):
        """Obtain syntactic category of a connective.

        Args:
        a_conn_toks (list):
          list of connective's tokens
        a_tree (dict):
          syntactic tree

        Returns:
        (str): syntactic category of the connective

        """
        inode_id = self._get_path(a_conn_toks, a_tree)
        return a_tree[inode_id].label()

    def _get_prnt_cat(self, a_conn_toks, a_tree):
        """Obtain syntactic category of connective's parent.

        Args:
        a_conn_toks (list):
          list of connective's tokens
        a_tree (dict):
          syntactic tree

        Returns:
        (str): syntactic category of the connective

        """
        inode_id = self._get_path(a_conn_toks, a_tree)
        if len(a_conn_toks) > 1 and len(inode_id) > 1:
            inode_id = inode_id[:-1]
        if len(inode_id) > 1:
            return a_tree[inode_id[:-1]].label()
        else:
            return DFLT_PRNT

    def _get_sib(self, a_conn_toks, a_tree, a_side):
        """Find common ancestor of two nodes

        Args:
        a_conn_toks (list):
          list of connective's tokens
        a_tree (dict):
          syntactic tree
        a_side (int):
          side of the sibling

        Returns:
        (str):
        label of the sibling (or empty string)

        """
        ret = "NONE"
        inode_id = self._get_path(a_conn_toks, a_tree)
        if not inode_id:
            return ret
        sib_path = [i for i in inode_id]
        if a_side == LEFT:
            if inode_id[-1] == 0:
                return ret
            sib_path[-1] -= 1
        else:
            if inode_id[-1] + 1 == len(a_tree[inode_id[:-1]]):
                return ret
            sib_path[-1] += 1
        return a_tree[sib_path].label()

    def _get_ctx_nodes(self, a_conn_toks, a_tree):
        """Find syntactic context nodes of the connective.

        Args:
        a_conn_toks (list):
          list of connective's tokens
        a_tree (dict):
          syntactic tree

        Returns:
        (list):
        list of context node labels

        """
        ret = []
        inode_id = self._get_path(a_conn_toks, a_tree)
        if len(inode_id) <= 1:
            return DFLT_PRNT
        prnt_node = a_tree[inode_id[:-1]]
        ret.append(prnt_node.label())
        for inode in prnt_node:
            ret.append(inode.label())
        return ret

    def _get_prev_conn(self, a_tok_id, a_tree):
        """Obtain  of the connective.

        Args:
        a_tok_id (int):
          index of the first token of the connective
        a_tree (dict):
          syntactic tree

        Returns:
        (list):
        list of context node labels

        """
        ret = []
        # obtain tokens of the sentence in question
        toks = [(i, t.lower()) for i, t in
                enumerate(a_tree.leaves()[:a_tok_id])]
        print("toks =", repr(toks))
        tokset = set(t[-1] for t in toks)
        sent_len = len(toks)
        # find matches of the first connective tokens
        matches = tokset & CONNTOKS
        if not matches:
            return ret
        print("matches =", repr(matches))
        # generate mapping from sentence tokens to their indices
        snt_tok2pos = defaultdict(list)
        for i, t in toks:
            snt_tok2pos[t].append(i)
        # generate mapping from connective parts to sentence indices
        conn_pos2snt_pos = defaultdict(list)
        found = False
        start_tok = None
        start = prev_start_pos = start_pos = -1
        # check each matched first token
        for imatch in matches:
            # iterate over each connective which starts with that token
            for i, iconn in CONNTOK2CONN[imatch]:
                if i != 0:
                    continue
                found = True
                start = prev_start_pos = -1
                # iterate over each separate part of that connective
                for jpart in iconn:
                    start_tok = jpart[0]
                    if start_tok not in snt_tok2pos:
                        found = False
                        break
                    # iterate over each token of that part
                    for start_pos in snt_tok2pos[start_tok]:
                        # make sure the next connective part starts after the
                        # previous one
                        if start_pos < prev_start_pos:
                            continue

                        if len(jpart) > sent_len - start_pos:
                            found = False
                            break
                        found = True
                        for ktok, snt_tok in zip(jpart, toks[start_pos:]):
                            if ktok != snt_tok[-1]:
                                found = False
                                break
                        if found:
                            break
                    if found:
                        if start == -1:
                            start = start_pos
                        prev_start_pos = start_pos
                    else:
                        break
                if found:
                    ret.append((start, iconn))
        ret.sort(key=lambda el: el[0])
        print("ret =", repr(ret))
        if len(matches) > 1:
            sys.exit(66)

    def _get_path(self, a_toks, a_tree):
        """Obtain path to the syntactic node covering all tokens.

        Args:
        a_toks (list):
          list of token(s)
        a_tree (dict):
          syntactic tree

        Returns:
        (tuple):
        path to the node covering all tokens

        """
        if len(a_toks) > 1:
            istart, iend = min(a_toks), max(a_toks) + 1
            return a_tree.treeposition_spanning_leaves(istart, iend)
        else:
            path = a_tree.leaf_treeposition(a_toks[0])
            if len(path) > 2:
                path = path[:-2]
            elif len(path) > 1:
                path = path[:-1]
            return path
