"""Implement Layer classes in Tensorflow"""
from __future__ import division
from __future__ import absolute_import

import logging

import tensorflow as tf

from ...base import layer as base_layer
from ...base import getter
from .. import scope, wrapper

__all__ = ['Dense']

_LG = logging.getLogger(__name__)


def _get_weight_init(config):
    config = config or {'typename': 'Xavier'}
    return getter.get_initializer(
        config['typename'])(**config.get('args', {}))


def _get_bias_init(config):
    config = config or {'typename': 'Constant', 'args': {'value': 0.1}}
    return getter.get_initializer(
        config['typename'])(**config.get('args', {}))


class Dense(base_layer.BaseDense):
    """Implement Dense layer in Tensorflow.

    See :any:`BaseDense` for detail.
    """
    def _build_weight(self, shape, dtype):
        init = _get_weight_init(self.args['initializers'].get('weight'))
        weight = scope.get_variable(
            name='weight', shape=shape, initializer=init, dtype=dtype)
        self._add_parameter('weight', weight)

    def _build_bias(self, shape, dtype):
        init = _get_bias_init(self.args['initializers'].get('bias'))
        bias = scope.get_variable(
            name='bias', shape=shape, initializer=init, dtype=dtype)
        self._add_parameter('bias', bias)

    def _instantiate_parameters(self, n_inputs, dtype):
        if 'weight' not in self._parameter_variables:
            shape = (n_inputs, self.args['n_nodes'])
            self._build_weight(shape=shape, dtype=dtype)

        if not self.args['with_bias']:
            return

        if 'bias' not in self._parameter_variables:
            shape = (self.args['n_nodes'],)
            self._build_bias(shape=shape, dtype=dtype)

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
