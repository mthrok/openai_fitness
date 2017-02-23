"""Implement Layer classes in Tensorflow"""
from __future__ import division
from __future__ import absolute_import

import logging

import tensorflow as tf

from ...base import layer as base_layer
from .. import scope, wrapper
from .common import get_initializers

__all__ = ['Dense']

_LG = logging.getLogger(__name__)


class Dense(base_layer.BaseDense):
    """Implement Dense layer in Tensorflow.

    See :any:`BaseDense` for detail.
    """
    def _instantiate_parameters(self, n_inputs, dtype):
        initializers = get_initializers(
            self.args.get('initializers') or {}, self.args['with_bias'])

        w_shape = (n_inputs, self.args['n_nodes'])
        weight = scope.get_variable(
            name='weight', shape=w_shape, dtype=dtype,
            initializer=initializers['weight'])
        self._add_parameter('weight', weight)

        if self.args['with_bias']:
            b_shape = (self.args['n_nodes'],)
            bias = scope.get_variable(
                name='bias', shape=b_shape, dtype=dtype,
                initializer=initializers['bias'])
            self._add_parameter('bias', bias)

    def _build(self, input_tensor):
        if not self._parameter_variables:
            self._instantiate_parameters(
                input_tensor.shape[1], input_tensor.dtype)

        weight = self._get_parameter('weight').unwrap()
        output = tf.matmul(input_tensor.unwrap(), weight)

        if self.args['with_bias']:
            bias = self._get_parameter('bias').unwrap()
            output = tf.add(output, bias, name='output')
        return wrapper.Tensor(output, name='output')
