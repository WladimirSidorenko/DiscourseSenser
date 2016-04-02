#!/usr//bin/env python
# -*- coding: utf-8; mode: python; -*-

"""Module providing common utils for discourse senser components.

Attributes:
timeit (class): parameterized decorator for measurinng time performance

"""

##################################################################
# Imports
from __future__ import print_function

from datetime import datetime
from functools import wraps
import sys


##################################################################
# Class
class timeit(object):
    """Decorator class for measuring time performance.

    Attributes:
    __init__: class constructor
    __call__: actual execution routine

    """

    def __init__(self, a_msg):
        """Class constructor.

        Args:
        a_msg (str): debug message to print

        """
        self.msg = a_msg

    def __call__(self, a_func):
        """Decorator function.

        Args:
        a_func (method): decorated method

        Returns:
        (method): wrapped method

        """
        def _wrapper(*args, **kwargs):
            print(self.msg + " started", file=sys.stderr)
            start_time = datetime.utcnow()
            a_func(*args, **kwargs)
            end_time = datetime.utcnow()
            time_delta = (end_time - start_time).total_seconds()
            print(self.msg + " finished ({:.2f} sec)".format(time_delta),
                  file=sys.stderr)
        return wraps(a_func)(_wrapper)

##################################################################
# Methods
