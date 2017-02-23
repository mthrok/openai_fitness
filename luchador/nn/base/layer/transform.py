"""Define common interface for Layer classes"""
from __future__ import division
from __future__ import absolute_import

import logging

from .base import BaseLayer

__all__ = ['BaseFlatten', 'BaseTile', 'BaseConcat']

_LG = logging.getLogger(__name__)

# pylint: disable=abstract-method


class BaseFlatten(BaseLayer):
    """Reshape 4D tensor into 2D tensor"""
    def __init__(self):
        super(BaseFlatten, self).__init__()


class BaseTile(BaseLayer):
    """Tile tensor"""
    def __init__(self, pattern):
        super(BaseTile, self).__init__(pattern=pattern)


class BaseConcat(BaseLayer):
    """Concatenate variables"""
    def __init__(self, axis=1):
        super(BaseConcat, self).__init__(axis=axis)
