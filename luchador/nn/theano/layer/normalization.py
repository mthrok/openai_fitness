"""Implement BatchNormalization Layer class in Theano"""
from __future__ import division
from __future__ import absolute_import

import logging
from collections import OrderedDict

import theano.tensor as T

from ...base import layer as base_layer
from .. import wrapper, initializer

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

        if self._parameter_variables['mean'] is None:
            mean = wrapper.get_variable(
                name='mean', shape=shape, trainable=False,
                initializer=initializer.Constant(0), dtype=dtype)
            self.set_parameter_variables(mean=mean)

        if self._parameter_variables['var'] is None:
            var = wrapper.get_variable(
                name='var', shape=shape, trainable=False,
                initializer=initializer.Constant(1), dtype=dtype)
            self.set_parameter_variables(var=var)

        if self._parameter_variables['scale'] is None:
            scale_val = self.args['scale']
            scale = wrapper.get_variable(
                name='scale', shape=shape, trainable=True,
                initializer=initializer.Constant(scale_val), dtype=dtype)
            self.set_parameter_variables(scale=scale)

        if self._parameter_variables['offset'] is None:
            offset_val = self.args['offset']
            offset = wrapper.get_variable(
                name='offset', shape=shape, trainable=True,
                initializer=initializer.Constant(offset_val), dtype=dtype)
            self.set_parameter_variables(offset=offset)

    def _build(self, input_tensor):
        self._instantiate_parameters(
            input_tensor.shape, input_tensor.dtype)

        input_tensor_ = input_tensor.unwrap()

        mean_acc = self.get_parameter_variables('mean').unwrap()
        var_acc = self.get_parameter_variables('var').unwrap()
        scale = self.get_parameter_variables('scale').unwrap()
        offset = self.get_parameter_variables('offset').unwrap()

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
