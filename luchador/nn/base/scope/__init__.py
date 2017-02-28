"""Initialize scoping module"""
from __future__ import absolute_import

import luchador

# pylint: disable=wildcard-import
if luchador.get_nn_backend() == 'tensorflow':
    from .tensorflow import *  # noqa
else:
    from .theano import *  # noqa
