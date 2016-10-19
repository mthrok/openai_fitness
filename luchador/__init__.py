from __future__ import absolute_import

from .version import __version__  # noqa: F401
from .logging import init_logger

init_logger('luchador')

from .configure import *  # noqa: F401, F403
