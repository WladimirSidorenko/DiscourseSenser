#!/usr/bin/env python
# -*- coding: utf-8; mode: python; -*-

"""Module providing class for Wang sense disambiguation.

Attributes:
WangExplicitSenser (class):
  class that predict senses of explicit relations

"""

##################################################################
# Imports
from __future__ import absolute_import, print_function

from dsenser.constants import CONNECTIVE, DOC_ID, \
    TOK_LIST, SENTENCES, WORDS, POS, TOK_IDX, SNT_ID, \
    PARSE_TREE
from dsenser.resources import CONNTOK2CONN, CONNTOKS, conn2str
from dsenser.wang.wangbase import WangBaseSenser
from dsenser.utils import timeit

from collections import defaultdict
from nltk import Tree
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

import re
import sys

##################################################################
# Variables and Constants
ENCODING = "utf-8"
PREV_NONE = "prev1_NONE"
SPACE_RE = re.compile(r"\s+")
MULTISPACE_RE = re.compile(r"\s\s+")
EQ_RE = re.compile(r"=+")
LEFT = 1
RIGHT = 2
DFLT_PRNT = "SBAR"
AS = "as"
WHEN = "when"
DFLT_C = 0.3


##################################################################
# Class
class WangExplicitSenser(WangBaseSenser):
    """Class for disambiguating explicit connectives.

    Attrs:
    n_y (int): number of distinct classes

    Methods:

    """

    def __init__(self, a_clf=None):
        """Class constructor.

        Args:
        a_clf (classifier or None):
        classifier to use or None for default

        """
        self.n_y = -1
        self.ctype = "explicit"
        classifier = a_clf or LinearSVC(C=DFLT_C, dual=False,
                                        multi_class="crammer_singer")
        self._model = Pipeline([('vectorizer', DictVectorizer()),
                                ('classifier', classifier)])

    @timeit("Training explicit Wang classifier...")
    def train(self, *args, **kwargs):
        super(WangExplicitSenser, self).train(*args, **kwargs)

    def _extract_features(self, a_rel, a_parses):
        """Extract classification features for a given relation.

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
        snt_id = a_rel[CONNECTIVE][TOK_LIST][-1][SNT_ID]

        ###########################
        # connective token features
        conn_tok = self._get_conn_txt(doc_id, a_rel, a_parses)
        conn_ltok = conn_tok.lower()
        assert conn_tok, \
            "Connective string cannot be empty for explicit classifier."
        conn_str = "ConStr-" + conn_tok
        feats[conn_str] = 1
        conn_lstr = conn_str.lower()
        feats[conn_lstr] = 1
        # obtain POS of the connective
        conn_pos = "conpos-" + self._get_conn_pos(doc_id, a_rel, a_parses)
        feats[conn_pos] = 1
        # obtain token preceding the connective
        prev_tok = "prevtok-" + self._get_conn_prev(doc_id, a_rel, a_parses)
        feats["Tok1Tok2-{:s}|{:s}".format(conn_str, prev_tok)] = 1
        ####################
        # syntactic features
        tree_str = a_parses[doc_id][SENTENCES][snt_id][PARSE_TREE]
        parse_tree = Tree.fromstring(tree_str)
        if not parse_tree.leaves():
            print("Invalid parse tree for sentence {:d}".format(snt_id),
                  file=sys.stderr)
            return {self._escape_feat(k): v for k, v in feats.iteritems()}
        conn_t_ids = [t[-1] for t in a_rel[CONNECTIVE][TOK_LIST]]
        scat = "SyntCat-" + self._get_cat(conn_t_ids, parse_tree)
        feats[scat] = 1
        prnt_cat = "PrntCat-" + self._get_prnt_cat(conn_t_ids, parse_tree)
        feats[prnt_cat] = 1
        left_sib = "LeftSib-" + self._get_sib(conn_t_ids, parse_tree, LEFT)
        feats[left_sib] = 1
        right_sib = "RightSib-" + self._get_sib(conn_t_ids, parse_tree, RIGHT)
        feats[right_sib] = 1

        ################
        # joint features
        feats[conn_lstr + '|' + scat] = 1
        feats[conn_lstr + '|' + prnt_cat] = 1
        feats[conn_lstr + '|' + left_sib] = 1
        feats[conn_lstr + '|' + right_sib] = 1

        feats[scat + '|' + prnt_cat] = 1
        feats[scat + '|' + left_sib] = 1
        feats[scat + '|' + right_sib] = 1
        feats[prnt_cat + '|' + left_sib] = 1
        feats[prnt_cat + '|' + right_sib] = 1
        feats[left_sib + '|' + right_sib] = 1
        #################
        # Wang's features
        ctx_nodes = "CtxNodes-" + '-'.join(self._get_ctx_nodes(conn_t_ids,
                                                               parse_tree))
        feats[ctx_nodes] = 1

        if conn_ltok == AS or conn_ltok == WHEN:
            toks = a_parses[doc_id][SENTENCES][snt_id][WORDS]
            prev_conn, prev_pos = self._get_prev_conn(conn_t_ids[0],
                                                      parse_tree,
                                                      toks)
            if prev_conn:
                feats[conn_ltok + '-' + conn2str(prev_conn[0])] = 1
                feats[conn_ltok + "PoS-" + ','.join(prev_pos[0])] = 1
        if conn_ltok != AS:
            feats["NotAs"] = 1
        if conn_ltok != WHEN:
            feats["NotWhen"] = 1
        ##########################
        # Normalize feature names
        return {self._escape_feat(k): v for k, v in feats.iteritems()}

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

    def _get_prev_conn(self, a_tok_id, a_tree, a_toks):
        """Obtain  of the connective.

        Args:
        a_tok_id (int):
          index of the first token of the connective
        a_tree (dict):
          syntactic tree
        a_toks (list):
          list of sentence tokens

        Returns:
        (2-tupe(list, list)):
          list of previous connective tokens and their PoS tags

        """
        ret_conn = []
        ret_pos = []
        # obtain tokens of the sentence in question
        toks = [(i, t.lower()) for i, t in
                enumerate(a_tree.leaves()[:a_tok_id])]
        tokset = set(t[-1] for t in toks)
        sent_len = len(toks)
        # find matches of the first connective tokens
        matches = tokset & CONNTOKS
        if not matches:
            return (ret_conn, ret_pos)
        # generate mapping from sentence tokens to their indices
        snt_tok2pos = defaultdict(list)
        for i, t in toks:
            snt_tok2pos[t].append(i)
        matches = [(ipos, imatch) for imatch in matches
                   for ipos in snt_tok2pos[imatch]]
        matches.sort(key=lambda el: el[0])
        # generate mapping from connective parts to sentence indices
        found = False
        pos_tags = None
        start_tok = None
        start = prev_start_pos = start_pos = -1
        # check each matched first token
        for istart, imatch in matches:
            # iterate over each connective which starts with that token
            for i, iconn in CONNTOK2CONN[imatch]:
                if i != 0:
                    continue
                found = True
                pos_tags = []
                start = -1
                prev_start_pos = istart
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
                            pos_tags.extend(a_toks[t_id][1][POS] for t_id in
                                            xrange(start_pos,
                                                   start_pos + len(jpart)))
                            break
                    if found:
                        if start == -1:
                            start = start_pos
                        prev_start_pos = start_pos
                    else:
                        break
                if found:
                    ret_conn.append((start, iconn))
                    ret_pos.append((start, pos_tags))
        # sort connectives and their pos tags according to the starting
        # position
        ret_conn.sort(key=lambda el: el[0])
        ret_pos.sort(key=lambda el: el[0])
        # return only connectives and their pos tags
        return ([el[-1] for el in ret_conn], [el[-1] for el in ret_pos])

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

    def _escape_feat(self, a_feat):
        """Replace characters that might confuse dict vectorize.

        Args:
        a_feat (str): feature to be escaped

        Return:
        (str):
        escaped feature

        """
        a_feat = MULTISPACE_RE.sub(' ', a_feat)
        return EQ_RE.sub('_', a_feat)
