"""Initialize scope module

Scoping is a fundamental mechanism which must be defined at the lowest level.
Therefore, unlike other backend module, this is placed under base module.
"""
from __future__ import absolute_import

import luchador

# pylint: disable=wildcard-import
if luchador.get_nn_backend() == 'tensorflow':
    from .tensorflow import *  # noqa
else:
    from .theano import *  # noqa
