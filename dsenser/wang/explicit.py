#!/usr/bin/env python
# -*- coding: utf-8; mode: python; -*-

"""Module providing class for Wang sense disambiguation.

Attributes:
  WangExplicitSenser (class):
    class that predict senses of explicit relations

"""

##################################################################
# Imports
from __future__ import absolute_import, print_function, unicode_literals

from dsenser.constants import ARG1, ARG2, CONNECTIVE, DOC_ID, \
    TOK_ID, TOK_LIST, SENTENCES, WORDS, POS, TOK_IDX, SNT_ID, \
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
NEGATION = ("not", "n't")
MODALITY = {"can": 0, "may": 1, "must": 2, "need": 3, "shall": 4,
            "will": 5, "could": 0, "would": 5, "might": 1,
            "should": 4, "'ll": 4, "wo": 5, "sha": 4, "ca": 0,
            "have to": 6, "had to": 6, "'d to": 6, "'ve to": 6
            }


##################################################################
# Class
class WangExplicitSenser(WangBaseSenser):
    """Class for disambiguating explicit connectives.

    Attributes:
      n_y (int): number of distinct classes

    """

    def __init__(self, a_clf=None):
        """Class constructor.

        Args:
          a_clf (classifier or None):
            classifier to use or None for default

        """
        self.n_y = -1
        self.ctype = "explicit"
        classifier = a_clf or LinearSVC(C=DFLT_C,
                                        loss="hinge",
                                        penalty="l1",
                                        dual=False,
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
          void:

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
        feats[conn_ltok + '|' + scat] = 1
        feats[conn_ltok + '|' + prnt_cat] = 1
        feats[conn_ltok + '|' + left_sib] = 1
        feats[conn_ltok + '|' + right_sib] = 1

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

        #########################
        # Own features (TS)

        toks_pos1 = self._get_toks_pos(a_parses[doc_id][SENTENCES],
                                       a_rel, ARG1)
        toks_pos2 = self._get_toks_pos(a_parses[doc_id][SENTENCES],
                                       a_rel, ARG2)
        # modality (copied from implicit)
        self._get_modality(feats, toks_pos1, toks_pos2)

        # negation
        self._get_negation(feats, toks_pos1, toks_pos2)

        # obtain token (and pos) following the connective
        (succ_tok, succ_pos) = self._get_conn_succ(doc_id, a_rel, a_parses)
        feats["succtok-" + succ_tok] = 1
        feats["succpos-" + succ_pos] = 1

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
          str: connective's text

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
          str: connective's part-of-speech tags

        """
        return '_'.join([
            a_parses[a_doc_id][SENTENCES][snt_id][WORDS][tok_id][1][POS]
            for _, _, _, snt_id, tok_id in a_rel[CONNECTIVE][TOK_LIST]])

    def _get_conn_prev(self, a_doc_id, a_rel, a_parses):
        """Obtain token preceding the connective.

        Args:
          a_doc_id (str):
            id of the document containing the connective
          a_rel (dict):
            discourse relation to extract features for
          a_parses (dict):
            parsed data

        Returns:
          str: connective's text

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

    def _get_conn_succ(self, a_doc_id, a_rel, a_parses):
        """Obtain the token and POS following the connective

        Args:
          a_doc_id (str):
            id of the document containing the connectiv
          a_rel (dict):
            discourse relation to extract features for
          a_parses (dict):
            parsed data

        Returns:
          tuple(str, pos): next token and pos

        """
        _, _, _, snt_id, tok_id = a_rel[CONNECTIVE][TOK_LIST][0]
        if len(a_parses[a_doc_id][SENTENCES][snt_id][WORDS]) > \
           (tok_id + 1):
            tok_id += 1
        elif len(a_parses[a_doc_id][SENTENCES]) > (snt_id + 1):
            snt_id += 1
            tok_id = 0
        else:
            return ("succ_none", "succ_tok_none")
        return (a_parses[a_doc_id][SENTENCES][snt_id][WORDS][tok_id][TOK_IDX],
                a_parses[a_doc_id][SENTENCES][snt_id][WORDS][tok_id][1][POS])

    def _get_cat(self, a_conn_toks, a_tree):
        """Obtain syntactic category of a connective.

        Args:
          a_conn_toks (list):
            list of connective's tokens
          a_tree (dict):
            syntactic tree

        Returns:
          str: syntactic category of the connective

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
          str: syntactic category of the connective

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
          str: label of the sibling (or empty string)

        """
        ret = "NONE"
        inode_id = self._get_path(a_conn_toks, a_tree)
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
          list: list of context node labels

        """
        ret = []
        inode_id = self._get_path(a_conn_toks, a_tree)
        if len(inode_id) <= 1:
            return [DFLT_PRNT]
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
          tuple:
            list of previous connective tokens and list of their PoS tags

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
          tuple: path to the node covering all tokens

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

    def _get_negation(self, a_feats, a_toks1, a_toks2):
        """Estimate polarity values of the given relation.

        Args:
          a_feats (dict):
            target feature dictionary
          a_toks1 (list(str)):
            list of tokens from the 1-st argument
          a_toks2 (list(str)):
            list of tokens from the 2-nd argument

        Returns:
          void: updates `a_feats` dictionary in place

        """
        neg1 = self._get_arg_negation(a_toks1)
        neg2 = self._get_arg_negation(a_toks2)
        #add negation features
        a_feats["Neg1-" + neg1] = 1
        a_feats["Neg2-" + neg2] = 1
        a_feats["JointNeg-" + neg1 + "|" + neg2] = 1

    def _get_arg_negation(self, a_toks):
        """Estimate polarity of the given relation argument.

        Args:
          a_toks (list): indices of argument's tokens

        Returns:
          list: argument's tokens

        """
        ret = "pos"
        bigram = None
        for i, itok in enumerate(a_toks):
            itok = itok[0]
            if itok in NEGATION:
                ret = "neg"
        return ret

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
          updates ``a_feats`` dictionary in place

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
          list(int):

        """
        ret = [0] * 7
        bigram = None
        max_i = len(a_toks) - 1
        for i, itok in enumerate(a_toks):
            itok = itok[0]
            if itok in MODALITY:
                ret[MODALITY[itok]] = 1
            if i < max_i:
                bigram = itok + ' ' + a_toks[i + 1][0]
                if bigram in MODALITY:
                    ret[MODALITY[bigram]] = 1
        return ret

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
          list:
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

    def _get_snt2tok(self, a_tok_list):
        """Generate mapping from sentence indices to token lists.

        Args:
          a_tok_list (list):
            list of sentence and token indices pertaining to the argument

        Returns:
          collections.defaultdict:
            mapping from sentence indices to token lists

        """
        snt2tok_pos = defaultdict(set)
        for el in a_tok_list:
            snt_id = el[SNT_ID]
            snt2tok_pos[snt_id].add(el[TOK_ID])
        return snt2tok_pos

    def _escape_feat(self, a_feat):
        """Replace characters that might confuse dict vectorize.

        Args:
          a_feat (str): feature to be escaped

        Return:
          str: escaped feature

        """
        a_feat = MULTISPACE_RE.sub(' ', a_feat).strip()
        return EQ_RE.sub('_', a_feat)
