"""Initialize Neural Network backend"""
from __future__ import absolute_import

import luchador

# pylint: disable=wildcard-import
if luchador.get_nn_backend() == 'tensorflow':
    from . import tensorflow as backend  # noqa
    from .tensorflow import *  # noqa
else:
    from . import theano as backend  # noqa
    from .theano import *  # noqa
