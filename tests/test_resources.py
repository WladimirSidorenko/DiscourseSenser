#!/usr/bin/env python
# -*- coding: utf-8; mode: python; -*-

##################################################################
# Imports
from __future__ import absolute_import

from dsenser.constants import CONNECTIVE, TOK_LIST
from dsenser.utils import singleton, is_explicit


##################################################################
# Auxiliary Class
@singleton
class Aux(object):
    m1 = 3


##################################################################
# Test Classes
def test_singleton():
    a1 = Aux
    a2 = Aux
    assert a1.m1 == 3
    assert a2.m1 == 3
    assert a1 is a2


def test_is_explicit():
    assert is_explicit({CONNECTIVE: {TOK_LIST: [1, 2, 3]}})
    assert is_explicit({CONNECTIVE: {TOK_LIST: []}}) is False
