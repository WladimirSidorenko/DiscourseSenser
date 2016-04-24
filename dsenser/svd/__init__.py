#!/usr//bin/env python
# -*- coding: utf-8; mode: python; -*-

"""Package providing modules for SVD sense classification.

Attributes:
__all__ (list): variables and modules provided for export
__version__ (str): version of this package
svd (module): main module providing SVDSenser
base (module): module providing base SVDSenser
explicit (module): module providing explicit SVDSenser
implicit (module): module providing implicit SVDSenser

"""

##################################################################
# Imports
from __future__ import absolute_import, unicode_literals

from dsenser.svd.svd import SVDSenser

##################################################################
# Variables and Constants
__all__ = ["SVDSenser"]
__version__ = "0.0.1"
