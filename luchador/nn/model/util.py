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
from luchador.nn.base import BaseWrapper

from .sequential import make_sequential_model


_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'data')
_LG = logging.getLogger(__name__)


def _get_input():
    for class_ in get_subclasses(BaseWrapper):
        if class_.__name__ == 'Input':
            return class_
    raise ValueError('`Input` class is not defined in current backend.')


def _make_input(input_config):
    if input_config['typename'] == 'Input':
        return _get_input()(**input_config['args'])


def make_model(model_config):
    """Make model from model configuration

    Parameters
    ----------
    model_config : dict
        model configuration.

    Returns
    -------
    Model
        Resulting model
    """
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
