"""Utility functions for configuring log"""
import logging
from luchador.util import initialize_logger as init_logger


def initialize_logger(debug):
    """Initialize logger"""
    format = (
        '%(asctime)s: %(levelname)5s: %(message)s' if not debug else 
        '%(asctime)s: %(levelname)5s: %(funcName)10s: %(message)s'
    )
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, format=format)
    init_logger(name='luchador', message_format=format, level=level)
