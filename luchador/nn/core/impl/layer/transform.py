"""Define common interface for Layer classes"""
from __future__ import division
from __future__ import absolute_import

from ...base import BaseLayer
from ...backend import layer
__all__ = ['Flatten', 'Tile', 'Concat']
# pylint: disable=abstract-method


class Flatten(layer.Flatten, BaseLayer):
    """Reshape 4D tensor into 2D tensor"""
    def __init__(self, name='Flatten'):
        super(Flatten, self).__init__(name=name)


class Tile(layer.Tile, BaseLayer):
    """Tile tensor"""
    def __init__(self, pattern, name='Tile'):
        super(Tile, self).__init__(pattern=pattern, name=name)


class Concat(layer.Concat, BaseLayer):
    """Concatenate variables"""
    def __init__(self, axis=1, name='Concat'):
        super(Concat, self).__init__(axis=axis, name=name)
