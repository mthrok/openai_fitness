"""Utility functions for facilitating model construction"""
from __future__ import absolute_import

import logging

from luchador.util import get_subclasses
from ..base import BaseInput
from ..model import Sequential
from .getter import get_input, get_layer, get_tensor

_LG = logging.getLogger(__name__)


def _get_input():
    """Get Input class"""
    for class_ in get_subclasses(BaseInput):
        if class_.__name__ == 'Input':
            return class_
    raise ValueError('`Input` class is not defined in current backend.')


def _make_input(config):
    if isinstance(config, list):
        return [_make_input(cfg) for cfg in config]

    type_ = config['typename']
    if type_ == 'Input':
        if config.get('reuse'):
            input_ = get_input(config['name'])
        else:
            input_ = _get_input()(**config['args'])
    elif type_ == 'Tensor':
        input_ = get_tensor(name=config['name'])
    else:
        raise ValueError('Unexpected Input type: {}'.format(type_))
    return input_


def _make_sequential_model(layer_configs):
    """Make model from model configuration

    Parameters
    ----------
    layer_config : list
        Layer configuration.

    Returns
    -------
    Model
        Resulting model
    """
    model = Sequential()
    for config in layer_configs:
        if 'typename' not in config:
            raise RuntimeError('Layer name is not given')
        args = config.get('args', {})

        _LG.debug('    Constructing: %s: %s', config['typename'], args)
        layer = get_layer(config['typename'])(**args)
        model.add_layer(layer=layer, scope=config.get('scope', ''))
    return model


def make_model(model_config):
    """Make model from model configuration

    Parameters
    ----------
    model_config : dict or list
        model configuration in dict or list of configurations.

    Returns
    -------
    Model or list of Model
        Resulting models
    """
    if isinstance(model_config, list):
        return [make_model(cfg) for cfg in model_config]

    _type = model_config['model_type']
    if _type == 'Sequential':
        model = _make_sequential_model(model_config['layer_configs'])
    else:
        raise ValueError('Unexpected model type: {}'.format(_type))

    if model_config.get('input'):
        input_ = _make_input(model_config['input'])
        model(input_)
    return model
