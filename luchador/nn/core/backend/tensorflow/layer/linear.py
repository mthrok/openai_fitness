"""Implement Layer classes in Tensorflow"""
from __future__ import division
from __future__ import absolute_import

import logging

import tensorflow as tf

from luchador.nn.core.base import fetch_initializer
from .. import wrapper

__all__ = ['Dense']
_LG = logging.getLogger(__name__)
# pylint:disable=no-member


def _get_weight_init(config):
    config = config or {'typename': 'XavierInitializer'}
    return fetch_initializer(config['typename'])(**config.get('args', {}))


def _get_bias_init(config):
    config = config or {
        'typename': 'ConstantInitializer', 'args': {'value': 0.1}}
    return fetch_initializer(config['typename'])(**config.get('args', {}))


class Dense(object):
    """Implement Dense layer in Tensorflow.

    See :any:`BaseDense` for detail.
    """
    def _build_weight(self, shape, dtype):
        init = _get_weight_init(self.args['initializers'].get('weight'))
        weight = wrapper.make_variable(
            name='weight', shape=shape, initializer=init, dtype=dtype)
        self.set_parameter_variables(weight=weight)

    def _build_bias(self, shape, dtype):
        init = _get_bias_init(self.args['initializers'].get('bias'))
        bias = wrapper.make_variable(
            name='bias', shape=shape, initializer=init, dtype=dtype)
        self.set_parameter_variables(bias=bias)

    def _instantiate_parameters(self, n_inputs, dtype):
        if self._parameter_variables['weight'] is None:
            shape = (n_inputs, self.args['n_nodes'])
            self._build_weight(shape=shape, dtype=dtype)

        if not self.args['with_bias']:
            return

        if self._parameter_variables['bias'] is None:
            shape = (self.args['n_nodes'],)
            self._build_bias(shape=shape, dtype=dtype)

    def _build(self, input_tensor):
        self._instantiate_parameters(input_tensor.shape[1], input_tensor.dtype)

        weight = self.get_parameter_variable('weight').unwrap()
        output = tf.matmul(input_tensor.unwrap(), weight)

        if self.args['with_bias']:
            bias = self.get_parameter_variable('bias').unwrap()
            output = tf.add(output, bias, name='output')
        return wrapper.Tensor(output, name='output')
