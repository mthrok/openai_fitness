"""Define common interface for Layer classes"""
from __future__ import division
from __future__ import absolute_import

from ...base import BaseLayer
from ...backend import layer
from ...backend import ops
__all__ = ['Flatten', 'Concat', 'Reshape']
# pylint: disable=abstract-method


class Flatten(layer.Flatten, BaseLayer):
    """Reshape 4D tensor into 2D tensor"""
    def __init__(self, scope='Flatten'):
        super(Flatten, self).__init__(scope=scope)


class Concat(layer.Concat, BaseLayer):
    """Concatenate multiple Tensors"""
    def __init__(self, axis=1, scope='Concat'):
        super(Concat, self).__init__(axis=axis, scope=scope)


class Reshape(BaseLayer):
    """Reshape variable"""
    def __init__(self, shape, scope='Reshape'):
        super(Reshape, self).__init__(shape=shape, scope=scope)

    def _build(self, input_tensor):
        shape, scope = self.args['shape'], self.args['scope']
        return ops.reshape(input_tensor, shape, name=scope)
