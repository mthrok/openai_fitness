"""Utility functions for facilitating model construction"""
from __future__ import absolute_import

import os
import logging

from luchador.util import load_config

__all__ = ['get_model_config']
_LG = logging.getLogger(__name__)


def get_model_config(filepath, **parameters):
    """Load model configurations from file

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
    if not os.path.isfile(filepath):
        raise ValueError(
            'Model definition file ({}) was not found.'.format(filepath))
    return load_config(filepath, **parameters)
