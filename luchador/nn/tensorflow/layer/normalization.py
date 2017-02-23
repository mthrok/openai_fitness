"""Implement Layer classes in Tensorflow"""
from __future__ import division
from __future__ import absolute_import

import logging

import tensorflow as tf

import luchador
from ...base import layer as base_layer
from .. import scope, wrapper, initializer

__all__ = ['BatchNormalization']

_LG = logging.getLogger(__name__)


class BatchNormalization(base_layer.BaseBatchNormalization):
    """Implement BatchNormalization in Tensorflow.

    See :any:`BaseBatchNormalization` for detail.
    """
    def _instantiate_parameters(self, input_shape):
        dim, fmt = len(input_shape), luchador.get_nn_conv_format()
        channel = 1 if dim == 2 or fmt == 'NCHW' else 3

        self._axes = tuple(i for i in range(dim) if not i == channel)
        shape = tuple(input_shape[i] for i in range(dim) if i == channel)

        mean = scope.get_variable(
            name='mean', shape=shape,
            initializer=initializer.Constant(0), trainable=False)
        var = scope.get_variable(
            name='var', shape=shape,
            initializer=initializer.Constant(1), trainable=False)

        scale = scope.get_variable(
            name='scale', shape=shape, trainable=True,
            initializer=initializer.Constant(self.args['scale']))
        offset = scope.get_variable(
            name='offset', shape=shape, trainable=True,
            initializer=initializer.Constant(self.args['offset']))

        self._add_parameter('mean', mean)
        self._add_parameter('var', var)
        self._add_parameter('scale', scale)
        self._add_parameter('offset', offset)

    def _build(self, input_tensor):
        input_shape = input_tensor.shape
        if not self._parameter_variables:
            self._instantiate_parameters(input_shape)

        input_ = input_tensor.unwrap()
        decay, epsilon = self.args['decay'], self.args['epsilon']

        mean_acc = self._get_parameter('mean').unwrap()
        var_acc = self._get_parameter('var').unwrap()
        scale = self._get_parameter('scale').unwrap()
        offset = self._get_parameter('offset').unwrap()

        if self.args['learn']:
            mean_in, var_in = tf.nn.moments(input_, self._axes)

            new_mean_acc = decay * mean_acc + (1 - decay) * mean_in
            new_var_acc = decay * var_acc + (1 - decay) * var_in

            self._update_operation = wrapper.Operation(
                op=[
                    tf.assign(mean_acc, new_mean_acc),
                    tf.assign(var_acc, new_var_acc)
                ],
                name='bn_update',
            )

            mean_acc = new_mean_acc
            var_acc = new_var_acc

        output = tf.nn.batch_normalization(
            x=input_, mean=mean_acc, variance=var_acc, offset=offset,
            scale=scale, variance_epsilon=epsilon)
        return wrapper.Tensor(output, name='output')
