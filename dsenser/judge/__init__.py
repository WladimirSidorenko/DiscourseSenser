#!/usr/bin/env python
# -*- coding: utf-8; mode: python; -*-

"""Package providing modules for judge classification.

Attributes:
__all__ (list): variables and modules provided for export
__version__ (str): version of this package
wang (module): main module providing Wang sense classifier
explicit (module): module providing sense classifier of explicit connectives
implicit (module): module providing sense classifier of implicit connectives

"""

##################################################################
# Imports
from __future__ import absolute_import, unicode_literals

from dsenser.judge.judge import Judge

##################################################################
# Variables and Constants
__all__ = ["Judge"]
__version__ = "0.0.1"
