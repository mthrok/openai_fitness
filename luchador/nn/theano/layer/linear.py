"""Implement Layer classes in Theano"""
from __future__ import division
from __future__ import absolute_import

import logging

import theano.tensor as T

from ...base import layer as base_layer
from .. import scope, wrapper
from .common import get_initializers

__all__ = ['Dense']

_LG = logging.getLogger(__name__)


class Dense(base_layer.BaseDense):
    """Implement Dense layer in Theano.

    See :any:`BaseDense` for detail.
    """
    def _instantiate_parameters(self, n_inputs, dtype):
        initializers = get_initializers(
            self.args.get('initializers') or {}, self.args['with_bias'])

        w_shape = (n_inputs, self.args['n_nodes'])
        w_init = initializers['weight']
        self._add_parameter('weight', scope.get_variable(
            name='weight', shape=w_shape, initializer=w_init, dtype=dtype))

        if self.args['with_bias']:
            b_shape = (self.args['n_nodes'],)
            b_init = initializers['bias']
            self._add_parameter('bias', scope.get_variable(
                name='bias', shape=b_shape, initializer=b_init, dtype=dtype))

    def _build(self, input_tensor):
        input_shape = input_tensor.shape

        if not len(input_shape) == 2:
            raise ValueError('Input tensor must be 2D. '
                             'Insted of {}'.format(len(input_shape)))

        if not self._parameter_variables:
            self._instantiate_parameters(input_shape[1], input_tensor.dtype)

        weight = self._get_parameter('weight').unwrap()
        output = T.dot(input_tensor.unwrap(), weight)

        if self.args['with_bias']:
            bias = self._get_parameter('bias').unwrap()
            output = output + bias
        output_shape = (input_shape[0], self.args['n_nodes'])
        return wrapper.Tensor(output, shape=output_shape, name='output')
