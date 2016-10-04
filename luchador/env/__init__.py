from __future__ import absolute_import

import logging

from .base import *  # noqa: F401, F403

try:
    from .ale import *  # noqa: F401, F403
except ImportError:
    logging.getLogger(__name__).exception('Failed to import ALE Environment.')

from .flappy_bird import *  # noqa: F401, F403
