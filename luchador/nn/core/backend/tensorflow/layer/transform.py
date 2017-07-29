"""Implement Layer classes in Tensorflow"""
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf

from ..wrapper import Tensor

__all__ = ['Flatten', 'Concat']
# pylint: disable=no-self-use,no-member


def _prod(vals):
    ret = 1
    for val in vals:
        ret *= val
    return ret


class Flatten(object):
    """Implement Flatten in Tensorflow.

    See :any:`BaseFlatten` for detail.
    """
    def _build(self, input_tensor):
        in_shape = input_tensor.shape
        n_nodes = int(_prod(in_shape[1:]))
        out_shape = (-1, n_nodes)
        output = tf.reshape(input_tensor.unwrap(), out_shape, 'output')
        return Tensor(output, name='output')


class Concat(object):
    """Implement Concat in Tensorflow.

    See :any:`BaseConcat` for detail.
    """
    def _build(self, var_list):
        values = [var.unwrap() for var in var_list]
        output = tf.concat(values, axis=self.args['axis'])
        return Tensor(output, name='output')
