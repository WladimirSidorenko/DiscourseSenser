#!/usr//bin/env python
# -*- coding: utf-8; mode: python; -*-

"""Package providing modules for shallow discourse parsing.

Attributes:
__all__ (list): variables and modules provided for export
__version__ (str): version of this package
constants (module): module containing package constants
scorer (module): module containing CoNLL evaluation scripts
senser (module): module containing methods for sense disambiguation

"""

##################################################################
# Imports
from __future__ import absolute_import

# from dparser.constants import DiscourseParser
from dparser.dparser import DiscourseParser

##################################################################
# Variables and Constamts
__all__ = ["DiscourseParser", "scorer", "senser"]
__version__ = "0.0.1-alpha"
