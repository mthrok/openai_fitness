"""Module to define utility functions used in luchador.nn module

This module is expected to be loaded before backend is loaded,
thus should not cause cyclic import.
"""
from __future__ import absolute_import

import os
import logging
import StringIO

import yaml

from luchador.util import get_subclasses
import luchador.nn

from .model.sequential import make_sequential_model

__all__ = ['make_model', 'get_model_config']

_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'data')
_LG = logging.getLogger(__name__)


def _get_input():
    for class_ in get_subclasses(luchador.nn.base.wrapper.BaseWrapper):
        if class_.__name__ == 'Input':
            return class_
    raise ValueError('`Input` class is not defined in current backend.')


def _make_input(config):
    if isinstance(config, list):
        return [_make_input(cfg) for cfg in config]

    type_ = config['typename']
    if type_ == 'Input':
        input_ = _get_input()(**config['args'])
    elif type_ == 'Tensor':
        input_ = luchador.nn.get_tensor(name=config['name'])
    else:
        raise ValueError('Unexpected Input type: {}'.format(type_))
    return input_


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
        model = make_sequential_model(model_config['layer_configs'])
    else:
        raise ValueError('Unexpected model type: {}'.format(_type))

    if model_config.get('input'):
        input_ = _make_input(model_config['input'])
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
