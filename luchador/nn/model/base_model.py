"""Define base network model structure and fetch method"""
from __future__ import absolute_import

import abc

import luchador.util

__all__ = ['BaseModel', 'get_model']


class BaseModel(object):  # pylint: disable=too-few-public-methods
    """Base Model class"""
    __metaclass__ = abc.ABCMeta


def get_model(name):
    """Get Model class by name

    Parameters
    ----------
    name : str
        Name of Model to get

    Returns
    -------
    type
        Class found with the given name

    Raises
    ------
    ValueError
        When Model class with the given name is not found
    """
    for class_ in luchador.util.get_subclasses(BaseModel):
        if class_.__name__ == name:
            return class_
    raise ValueError('Unknown model: {}'.format(name))
