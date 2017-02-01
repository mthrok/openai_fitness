"""Module to define utility functions used in luchador.nn module

This module is expected to be loaded before other modules are loaded,
thus should not cause cyclic import.
"""
from __future__ import absolute_import

import os
import logging
import StringIO

import yaml

from luchador.util import get_subclasses
from luchador.nn import get_layer
from luchador.nn.base.wrapper import BaseTensor
from .model import get_model

_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'data')
_LG = logging.getLogger(__name__)


def _get_input():
    for class_ in get_subclasses(BaseTensor):
        if class_.__name__ == 'Input':
            return class_
    raise ValueError('`Input` class is not defined in cullent backend.')


def make_model(model_config):
    """Make model from model configuration

    Parameters
    ----------
    model_condig : JSON-compatible object
        model configuration.

    Returns
    -------
    Model
        Resulting model
    """
    model = get_model(model_config['model_type'])()
    for cfg in model_config['layer_configs']:
        layer_cfg = cfg['layer']
        if 'name' not in layer_cfg:
            raise RuntimeError('Layer name is not given')
        args = layer_cfg.get('args', {})
        _LG.debug('    Constructing: %s: %s', layer_cfg['name'], args)
        layer = get_layer(layer_cfg['name'])(**args)
        model.add_layer(layer=layer, scope=cfg.get('scope', ''))
    if 'input' in model_config:
        input_ = _get_input()(**model_config['input'])
        model(input_)
    return model


def get_model_config(model_name, **parameters):
    """Load pre-defined model configurations

    Parameters
    ----------
    model_name : str
        Model name or path to YAML file

    parameters
        Parameter for model config

    Returns
    -------
    JSON-compatible object
        Model configuration.
    """
    file_name = model_name
    if not file_name.endswith('.yml'):
        file_name = '{}.yml'.format(file_name)

    if os.path.isfile(file_name):
        file_path = file_name
    elif os.path.isfile(os.path.join(_DATA_DIR, file_name)):
        file_path = os.path.join(_DATA_DIR, file_name)
    else:
        raise ValueError(
            'No model definition file ({}) found.'.format(file_name))

    with open(file_path, 'r') as file_:
        model_text = file_.read()

    if parameters:
        model_text = model_text.format(**parameters)

    model_text = StringIO.StringIO(model_text)
    return yaml.load(model_text)