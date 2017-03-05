"""Utility functions for facilitating model construction"""
from __future__ import absolute_import

import os
import logging

from luchador.util import load_config
__all__ = ['get_model_config']

_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'data')
_LG = logging.getLogger(__name__)


def get_model_config(model_name, **parameters):
    """Load model configurations from library or file

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
            'Model definition file ({}) was not found.'.format(file_name))

    return load_config(file_path, **parameters)
