"""Implement Layer classes in Tensorflow"""
from __future__ import division
from __future__ import absolute_import

import logging

import tensorflow as tf

from luchador.nn.core.base import fetch_initializer
from ..wrapper import Tensor, make_variable

__all__ = ['ReLU', 'LeakyReLU', 'Softplus', 'Sigmoid', 'Tanh', 'Softmax']
_LG = logging.getLogger(__name__)
# pylint: disable=no-self-use, no-member


class ReLU(object):
    """Implement ReLU in Tensorflow.

    See :any:`BaseReLU` for detail.
    """
    def _build(self, input_tensor):
        output = tf.nn.relu(input_tensor.unwrap(), 'ouptut')
        return Tensor(output, name='output')


class LeakyReLU(object):
    """Implement LeakyReLU layer in Tensorflow.

    See :any:`LeakyReLU` for detail.
    """
    def _get_alpha(self):
        _alpha = self.args['alpha']
        initializer = fetch_initializer('ConstantInitializer')(value=_alpha)
        alpha = make_variable(name='alpha', shape=[], initializer=initializer)
        self._create_parameter_slot(
            'alpha', val=alpha, train=True, serialize=True)
        return alpha.unwrap()

    def _build(self, input_tensor):
        x = input_tensor.unwrap()
        alpha = self._get_alpha() if self.args['train'] else self.args['alpha']
        f1 = 0.5 * (1 + alpha)
        f2 = 0.5 * (1 - alpha)
        output = f1 * x + f2 * abs(x)
        return Tensor(output, name='output')


class Softplus(object):
    """Implement Softplus in Tensorflow.

    See :any:`BaseSoftplus` for detail.
    """
    def _build(self, input_tensor):
        output = tf.nn.softplus(input_tensor.unwrap())
        return Tensor(output, name='output')


class Sigmoid(object):
    """Implement Sigmoid in Tensorflow.

    See :any:`BaseSigmoid` for detail.
    """
    def _build(self, input_tensor):
        output = tf.sigmoid(input_tensor.unwrap(), 'output')
        return Tensor(output, name='output')


class Tanh(object):
    """Implement Tanh in Tensorflow.

    See :any:`BaseTanh` for detail.
    """
    def _build(self, input_tensor):
        output = tf.tanh(input_tensor.unwrap(), 'output')
        return Tensor(output, name='output')


class Softmax(object):
    """Implement Softmax in Tensorflow.

    See :any:`BaseSoftmax` for detail.
    """
    def _build(self, input_tensor):
        output = tf.nn.softmax(input_tensor.unwrap())
        return Tensor(output, name='output')
