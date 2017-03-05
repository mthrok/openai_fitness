"""Utility functions for facilitating model construction"""
from __future__ import absolute_import

import logging
from collections import OrderedDict

_LG = logging.getLogger(__name__)


class ConfigDict(OrderedDict):
    """Class for distinguishing ordinal dict and model config"""
    pass


def parse_config(config):
    """Mark valid configurations as ConfigDict class

    Parameters
    ----------
    config : object
        If config is dict subclass and contains `typename` key,
        that instance will be converted to `ConfigDict` class.
        As :any:`ConfigDict` is merely a subclass of dict, the structure
        of the input object does not change.

    Returns
    -------
    object
        If [parts of] Input object is dict with key ``typename``, they are
        converted to :any:`ConfigDict` (just a subclass of ``dict``).
        The input structure is parsed revursively, so if input value contains
        nested dict with key ``typename``, all of them are converted to
        :any:`ConfigDict`.
    """
    if isinstance(config, list):
        return [parse_config(cfg) for cfg in config]
    if isinstance(config, ConfigDict):
        return config
    if isinstance(config, dict):
        _config = ConfigDict() if 'typename' in config else OrderedDict()
        for key, value in config.items():
            _config[key] = parse_config(value)
        return _config
    return config
