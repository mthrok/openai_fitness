from __future__ import absolute_import

import logging

import luchador

from .base import *  # noqa: F401, F403

_LG = logging.getLogger(__name__)
_LG.info('Luchador Version: %s', luchador.__version__)
_LG.info('Luchador NN backend: %s', luchador.get_nn_backend())

if luchador.get_nn_backend() == 'tensorflow':
    from .tensorflow import *  # noqa: F401, F403
else:
    from .theano import *  # noqa: F401, F403
