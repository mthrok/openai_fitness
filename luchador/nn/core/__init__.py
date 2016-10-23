from __future__ import absolute_import

import logging

import luchador


logging.getLogger(__name__).info(
    'Using %s backend', luchador.get_nn_backend()
)

if luchador.get_nn_backend() == 'tensorflow':
    from .tensorflow import *  # noqa: F401, F403
else:
    from .theano import *  # noqa: F401, F403
