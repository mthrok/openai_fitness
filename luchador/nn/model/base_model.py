"""Define base network model structure and fetch method"""
from __future__ import absolute_import

import logging
from collections import OrderedDict

from luchador.util import fetch_subclasses
from ..core import get_variable_scope

__all__ = ['BaseModel', 'fetch_model', 'get_model']
_LG = logging.getLogger(__name__)
_MODELS = OrderedDict()


def _register(name, model):
    if name in _MODELS:
        _LG.warning('Model `%s` already exists.', name)
    _MODELS[name] = model


class BaseModel(object):  # pylint: disable=too-few-public-methods
    """Base Model class"""
    def __init__(self, name=None):
        super(BaseModel, self).__init__()
        scope = get_variable_scope().name
        if scope:
            name = '{}/{}'.format(scope, name)
        self.name = name
        self.input = None
        self.output = None

        if name:
            _register(name, self)


def fetch_model(name):
    """Get ``Model`` class by name

    Parameters
    ----------
    name : str
        Name of ``Model`` to get

    Returns
    -------
    type
        ``Model`` type found

    Raises
    ------
    ValueError
        When ``Model`` class with the given name is not found
    """
    for class_ in fetch_subclasses(BaseModel):
        if class_.__name__ == name:
            return class_
    raise ValueError('Unknown model: {}'.format(name))


def get_model(name):
    """Get an instance of ``Model``

    Parameters
    ----------
    name : str
        name of ``Model`` instance to get

    Returns
    -------
    Model
    """
    try:
        scope = get_variable_scope().name
        name_ = '{}/{}'.format(scope, name) if scope else name
        return _MODELS[name_]
    except KeyError:
        pass
    return _MODELS[name]
