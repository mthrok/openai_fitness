"""Utility functions for facilitating model construction"""
from __future__ import absolute_import

import logging

from ... import core
from .io import make_io_node

_LG = logging.getLogger(__name__)


def make_layer(layer_config):
    """Make Layer instance

    Parameters
    ----------
    layer_config : dict
        typename : str
            Name of Layer class to instanciate
        args : dict
            Constructor arguments for the Layer class
        parameters: dict
            Parameter Variables to resuse. Keys are parameter keys
            and values are configuration acceptable by :func:`make_io_node`.

    Returns
    -------
    layer
        Layer object
    """
    _LG.info('  Constructing Layer: %s', layer_config)
    if 'typename' not in layer_config:
        raise RuntimeError('Layer `typename` is not given')

    layer = core.fetch_layer(
        layer_config['typename'])(**layer_config.get('args', {}))

    if 'parameters' in layer_config:
        parameters = make_io_node(layer_config['parameters'])
        layer.set_parameter_variables(**parameters)
    return layer
