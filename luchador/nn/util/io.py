"""Utility functions for facilitating model construction"""
from __future__ import absolute_import

import os
import logging
import StringIO

import ruamel.yaml as yaml

import luchador.util

__all__ = ['get_model_config']


_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'data')
_LG = logging.getLogger(__name__)


def _convert_to_str(value):
    """Convert object into one-line YAML string"""
    if isinstance(value, str):
        return value
    if value is None:
        return 'null'

    if isinstance(value, dict):
        return '{{{}}}'.format(
            ', '.join(
                '{}: {}'.format(key, _convert_to_str(val))
                for key, val in value.items()
            )
        )

    if luchador.util.is_iteratable(value):
        return '[{}]'.format(
            ', '.join([_convert_to_str(val) for val in value]))
    return str(value)


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
        _params = {
            key: _convert_to_str(val) for key, val in parameters.items()}
        model_text = model_text.format(**_params)

    model_text = StringIO.StringIO(model_text)
    return yaml.safe_load(model_text)
