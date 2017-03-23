"""Initialize Neural Network module and load backend"""
from __future__ import absolute_import

import logging

# pylint: disable=wildcard-import
import luchador
from .core import *  # noqa
from .util import *  # noqa
from .saver import Saver  # noqa
from .summary import SummaryWriter  # noqa
from .model import get_model, fetch_model  # noqa

_LG = logging.getLogger(__name__)
_LG.info('Luchador Version: %s', luchador.__version__)
_LG.info('Luchador NN backend: %s', luchador.get_nn_backend())
