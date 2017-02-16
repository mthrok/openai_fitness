"""Define base network model structure and fetch method"""
from __future__ import absolute_import

import abc

import luchador.util

__all__ = ['BaseModel', 'get_model']


class BaseModel(object):  # pylint: disable=too-few-public-methods
    """Base Model class"""
    __metaclass__ = abc.ABCMeta
