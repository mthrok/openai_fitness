"""Initialize standard logger used in luchador"""
from __future__ import absolute_import

import logging

__all__ = ['initialize_logger']


def initialize_logger(name, message_format, level, propagate=False):
    """Initialize luchador logger"""
    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(message_format)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = propagate
    return logger
