"""Implement Layer classes in Tensorflow"""
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf

from ..wrapper import Tensor

__all__ = [
    'Flatten', 'Tile', 'Concat',
]
# pylint: disable=no-self-use,no-member


class Flatten(object):
    """Implement Flatten in Tensorflow.

    See :any:`BaseFlatten` for detail.
    """
    def _build(self, input_tensor):
        in_shape = input_tensor.shape
        n_nodes = reduce(lambda prod, dim: prod*dim, in_shape[1:], 1)
        out_shape = (-1, n_nodes)
        output = tf.reshape(input_tensor.unwrap(), out_shape, 'output')
        return Tensor(output, name='output')


class Tile(object):
    """Implement Tile layer in Tensorflow

    See :any:`BaseFlatten` for detail.
    """
    def _build(self, input_tensor):
        return input_tensor.tile(self.args['pattern'], name='output')


class Concat(object):
    """Implement Concat in Tensorflow.

    See :any:`BaseConcat` for detail.
    """
    def _build(self, var_list):
        values = [var.unwrap() for var in var_list]
        output = tf.concat(values, axis=self.args['axis'])
        return Tensor(output, name='output')
