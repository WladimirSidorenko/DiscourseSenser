#!/usr/bin/env python
# -*- coding: utf-8; mode: python; -*-

"""Module providing common utils for discourse senser components.

"""

##################################################################
# Imports
from __future__ import absolute_import, print_function

from dsenser.constants import CONNECTIVE, TOK_LIST

from datetime import datetime
from functools import wraps
import sys


##################################################################
# decorators
def singleton(cls):
    """Make `cls` instance unique across all calls.

    Args:
      cls (class):
        class to be decorated

    Retuns:
      object:
        singleton instance of the decorated class

    """
    instance = cls()
    instance.__call__ = lambda: instance
    return instance


class timeit(object):
    """Decorator class for measuring time performance.

    Attributes:
      msg (str): message to be printed on method invocation

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
          method: wrapped method

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
def is_explicit(a_rel):
    """Check whether given relation is explicit.

    Args:
      a_rel (dict):
       discourse relation to classify

    Returns:
      bool:
        ``True`` if the relation is explicit, ``False`` otherwise

    """
    return bool(a_rel[CONNECTIVE][TOK_LIST])
