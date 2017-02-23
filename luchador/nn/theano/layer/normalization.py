"""Implement BatchNormalization Layer class in Theano"""
from __future__ import division
from __future__ import absolute_import

import logging
from collections import OrderedDict

import theano.tensor as T

from ...base import layer as base_layer
from .. import scope, wrapper, initializer

__all__ = ['BatchNormalization']

_LG = logging.getLogger(__name__)


class BatchNormalization(base_layer.BaseBatchNormalization):
    """Implement BN layer in Theano.

    See :any:`BaseBatchNormalization` for detail.
    """
    def _instantiate_parameters(self, input_shape, dtype):
        dim = len(input_shape)
        shape = tuple(input_shape[i] for i in range(dim) if i == 1)
        self._axes = tuple(i for i in range(dim) if not i == 1)
        self._pattern = tuple((0 if i == 1 else 'x') for i in range(dim))

        _LG.debug('    Shape: %s', shape)
        _LG.debug('     Axes: %s', self._axes)
        _LG.debug('  Pattern: %s', self._pattern)

        mean = scope.get_variable(
            name='mean', shape=shape, trainable=False,
            initializer=initializer.Constant(0), dtype=dtype)
        var = scope.get_variable(
            name='var', shape=shape, trainable=False,
            initializer=initializer.Constant(1), dtype=dtype)

        scale = scope.get_variable(
            name='scale', shape=shape, trainable=True,
            initializer=initializer.Constant(self.args['scale']), dtype=dtype)
        offset = scope.get_variable(
            name='offset', shape=shape, trainable=True,
            initializer=initializer.Constant(self.args['offset']), dtype=dtype)

        self._add_parameter('mean', mean)
        self._add_parameter('var', var)
        self._add_parameter('scale', scale)
        self._add_parameter('offset', offset)

    def _build(self, input_tensor):
        if not self._parameter_variables:
            self._instantiate_parameters(
                input_tensor.shape, input_tensor.dtype)

        input_tensor_ = input_tensor.unwrap()

        mean_acc = self._get_parameter('mean').unwrap()
        var_acc = self._get_parameter('var').unwrap()
        scale = self._get_parameter('scale').unwrap()
        offset = self._get_parameter('offset').unwrap()

        if self.args['learn']:
            decay = self.args['decay']
            mean_in = input_tensor_.mean(axis=self._axes)
            var_in = input_tensor_.var(self._axes)

            new_mean_acc = decay * mean_acc + (1 - decay) * mean_in
            new_var_acc = decay * var_acc + (1 - decay) * var_in

            self._update_operation = wrapper.Operation(
                op=OrderedDict(
                    [(mean_acc, new_mean_acc), (var_acc, new_var_acc)]),
                name='bn_update',
            )

            mean_acc = new_mean_acc
            var_acc = new_var_acc

        mean_acc = mean_acc.dimshuffle(self._pattern)
        var_acc = var_acc.dimshuffle(self._pattern)
        scale = scale.dimshuffle(self._pattern)
        offset = offset.dimshuffle(self._pattern)

        stdi = T.inv(T.sqrt(var_acc + self.args['epsilon']))
        output = scale * (input_tensor_ - mean_acc) * stdi + offset
        return wrapper.Tensor(output, shape=input_tensor.shape, name='output')
