"""Utility functions for facilitating model construction"""
from __future__ import absolute_import

import logging
from collections import OrderedDict

from ...model import Sequential, Container
from .common import ConfigDict, parse_config
from .io import make_io_node
from .layer import make_layer

_LG = logging.getLogger(__name__)


def _make_sequential_model(layer_configs, input_config=None):
    """Make Sequential model instance from model configuration

    Parameters
    ----------
    layer_configs : list
        ``Layer`` configuration.

    input_config : dict or None
        ``Input`` configuraiton to the model. If give, layers are built on the
        input spcified by this configuration, otherwise, model is returned
        unbuilt.

    Returns
    -------
    Model
        Resulting model
    """
    model = Sequential()
    if input_config:
        tensor = make_io_node(input_config)
    for config in layer_configs:
        layer = make_layer(config)
        if input_config:
            tensor = layer(tensor)
        model.add_layer(layer)
    return model


def _make_container_model(input_config, model_configs, output_config=None):
    """Make ``Container`` model from model configuration

    Parameters
    ----------
    input_config : [list of] dict
        See :any::make_io_node

    model_config : list
        Model configuration.

    output_config : [list of] dict
        See :any::make_io_node

    Returns
    -------
    Model
        Resulting model
    """
    model = Container()
    model.input = make_io_node(input_config)
    for conf in model_configs:
        _LG.info('Building Model: %s', conf.get('name', 'No name defined'))
        model.add_model(conf['name'], make_model(conf))

    if output_config:
        model.output = make_io_node(output_config)
    return model


def _make_model(model_config):
    _type = model_config.get('typename', 'No model type found')
    if _type == 'Sequential':
        return _make_sequential_model(**model_config.get('args', {}))
    if _type == 'Container':
        return _make_container_model(**model_config.get('args', {}))
    raise ValueError('Unexpected model type: {}'.format(_type))


def _make_model_recursively(model_config):
    if isinstance(model_config, ConfigDict):
        return _make_model(model_config)
    if isinstance(model_config, list):
        return [_make_model_recursively(cfg) for cfg in model_config]
    if isinstance(model_config, dict):
        ret = OrderedDict()
        for key, value in model_config.items():
            ret[key] = _make_model_recursively(value)
        return ret

    raise ValueError('Invalid model config: {}'.format(model_config))


def make_model(model_config):
    """Make model from model configuration

    Parameters
    ----------
    model_config : [list of] dict
        Model configuration in dict or list of configurations.

    Returns
    -------
    [list of] Model
        Resulting model[s]
    """
    model_config = parse_config(model_config)
    return _make_model_recursively(model_config)
