from __future__ import absolute_import

import os
import logging


if os.environ.get('LUCHADOR_DEBUG'):
    _DEFAULT_LEVEL = logging.DEBUG
    _DEFAULT_MSG = '%(asctime)s: %(levelname)5s: %(funcName)10s: %(message)s'
else:
    _DEFAULT_LEVEL = logging.INFO
    _DEFAULT_MSG = '%(asctime)s: %(levelname)5s: %(message)s'


def init_logger(
        name,
        msg_fmt=_DEFAULT_MSG,
        level=_DEFAULT_LEVEL,
        propagate=False):
    """Initialize luchador logger"""
    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(msg_fmt)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = propagate
    return logger
