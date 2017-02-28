"""Implement Layer classes in Tensorflow"""
from __future__ import division
from __future__ import absolute_import

import logging

import tensorflow as tf

from ...base import layer as base_layer
from ..wrapper import Tensor
from .common import LayerMixin

__all__ = [
    'ReLU', 'Softplus',
    'Sigmoid', 'Tanh',
    'Softmax',
]

_LG = logging.getLogger(__name__)


class ReLU(LayerMixin, base_layer.BaseReLU):
    """Implement ReLU in Tensorflow.

    See :any:`BaseReLU` for detail.
    """
    def _build(self, input_tensor):
        output = tf.nn.relu(input_tensor.unwrap(), 'ouptut')
        return Tensor(output, name='output')


class Softplus(LayerMixin, base_layer.BaseSoftplus):
    """Implement Softplus in Tensorflow.

    See :any:`BaseSoftplus` for detail.
    """
    def _build(self, input_tensor):
        output = tf.nn.softplus(input_tensor.unwrap())
        return Tensor(output, name='output')


class Sigmoid(LayerMixin, base_layer.BaseSigmoid):
    """Implement Sigmoid in Tensorflow.

    See :any:`BaseSigmoid` for detail.
    """
    def _build(self, input_tensor):
        output = tf.sigmoid(input_tensor.unwrap(), 'output')
        return Tensor(output, name='output')


class Tanh(LayerMixin, base_layer.BaseTanh):
    """Implement Tanh in Tensorflow.

    See :any:`BaseTanh` for detail.
    """
    def _build(self, input_tensor):
        output = tf.tanh(input_tensor.unwrap(), 'output')
        return Tensor(output, name='output')


class Softmax(LayerMixin, base_layer.BaseSoftmax):
    """Implement Softmax in Tensorflow.

    See :any:`BaseSoftmax` for detail.
    """
    def _build(self, input_tensor):
        output = tf.nn.softmax(input_tensor.unwrap())
        return Tensor(output, name='output')
