"""Implement Layer classes in Tensorflow"""
from __future__ import division
from __future__ import absolute_import

import logging

import tensorflow as tf

import luchador
from luchador.nn.core.base import get_initializer
from .. import wrapper

__all__ = ['BatchNormalization']
_LG = logging.getLogger(__name__)
# pylint: disable=no-self-use,no-member,attribute-defined-outside-init


class BatchNormalization(object):
    """Implement BatchNormalization in Tensorflow.

    See :any:`BaseBatchNormalization` for detail.
    """
    def _instantiate_parameters(self, input_shape):
        dim, fmt = len(input_shape), luchador.get_nn_conv_format()
        channel = 1 if dim == 2 or fmt == 'NCHW' else 3

        self._axes = tuple(i for i in range(dim) if not i == channel)
        shape = tuple(input_shape[i] for i in range(dim) if i == channel)

        const_init = get_initializer('ConstantInitializer')
        if self.get_parameter_variable('mean') is None:
            mean = wrapper.make_variable(
                name='mean', shape=shape,
                initializer=const_init(0), trainable=False)
            self.set_parameter_variables(mean=mean)

        if self.get_parameter_variable('var') is None:
            var = wrapper.make_variable(
                name='var', shape=shape,
                initializer=const_init(1), trainable=False)
            self.set_parameter_variables(var=var)

        if self.get_parameter_variable('scale') is None:
            scale = wrapper.make_variable(
                name='scale', shape=shape, trainable=True,
                initializer=const_init(self.args['scale']))
            self.set_parameter_variables(scale=scale)

        if self.get_parameter_variable('offset') is None:
            offset = wrapper.make_variable(
                name='offset', shape=shape, trainable=True,
                initializer=const_init(self.args['offset']))
            self.set_parameter_variables(offset=offset)

    def _build(self, input_tensor):
        input_shape = input_tensor.shape
        self._instantiate_parameters(input_shape)

        input_ = input_tensor.unwrap()
        decay, epsilon = self.args['decay'], self.args['epsilon']

        mean_acc = self.get_parameter_variable('mean').unwrap()
        var_acc = self.get_parameter_variable('var').unwrap()
        scale = self.get_parameter_variable('scale').unwrap()
        offset = self.get_parameter_variable('offset').unwrap()

        if self.args['learn']:
            mean_in, var_in = tf.nn.moments(input_, self._axes)

            new_mean_acc = decay * mean_acc + (1 - decay) * mean_in
            new_var_acc = decay * var_acc + (1 - decay) * var_in

            self._update_operations.append(
                wrapper.Operation(
                    op=tf.assign(mean_acc, new_mean_acc),
                    name='update_mean',
                )
            )
            self._update_operations.append(
                wrapper.Operation(
                    op=tf.assign(var_acc, new_var_acc),
                    name='update_var',
                )
            )
            mean_acc = new_mean_acc
            var_acc = new_var_acc

        output = tf.nn.batch_normalization(
            x=input_, mean=mean_acc, variance=var_acc, offset=offset,
            scale=scale, variance_epsilon=epsilon)
        return wrapper.Tensor(output, name='output')
