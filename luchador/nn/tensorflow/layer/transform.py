"""Implement Layer classes in Tensorflow"""
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf

from ...base import layer as base_layer
from ..wrapper import Tensor

__all__ = [
    'Flatten', 'Tile', 'Concat',
]


class Flatten(base_layer.BaseFlatten):
    """Implement Flatten in Tensorflow.

    See :any:`BaseFlatten` for detail.
    """
    def _build(self, input_tensor):
        in_shape = input_tensor.shape
        n_nodes = reduce(lambda prod, dim: prod*dim, in_shape[1:], 1)
        out_shape = (-1, n_nodes)
        output = tf.reshape(input_tensor.unwrap(), out_shape, 'output')
        return Tensor(output, name='output')


class Tile(base_layer.BaseTile):
    """Implement Tile layer in Tensorflow

    See :any:`BaseFlatten` for detail.
    """
    def _build(self, input_tensor):
        return input_tensor.tile(self.args['pattern'], name='output')


class Concat(base_layer.BaseConcat):
    """Implement Concat in Tensorflow.

    See :any:`BaseConcat` for detail.
    """
    def _build(self, var_list):
        values = [var.unwrap() for var in var_list]
        output = tf.concat(values, axis=self.args['axis'])
        return Tensor(output, name='output')
