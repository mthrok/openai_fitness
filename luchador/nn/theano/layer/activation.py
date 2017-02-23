"""Implement Activation Layers in Theano"""
from __future__ import division
from __future__ import absolute_import

import theano.tensor as T

from ...base import layer as base_layer
from ..wrapper import Tensor

__all__ = [
    'ReLU', 'Softplus',
    'Sigmoid', 'Tanh',
    'Softmax',
]


class ReLU(base_layer.BaseReLU):
    """Implement ReLU layer in Theano.

    See :any:`BaseReLU` for detail.
    """
    def _build(self, input_tensor):
        """Build rectified linear activation operation on input tensor"""
        input_shape = input_tensor.shape
        output_tensor = T.nnet.relu(input_tensor.unwrap())
        return Tensor(output_tensor, shape=input_shape, name='output')


class Softplus(base_layer.BaseSoftplus):
    """Implemente Softplus layer in Theano.

    See :any:`BaseSoftplus` for detail.
    """
    def _build(self, input_tensor):
        input_shape = input_tensor.shape
        output_tensor = T.nnet.softplus(input_tensor.unwrap())
        return Tensor(output_tensor, shape=input_shape, name='output')


class Sigmoid(base_layer.BaseSigmoid):
    """Implement Sigmoid layer in Theano.

    See :any:`BaseSigmoid` for detail.
    """
    def _build(self, input_tensor):
        input_shape = input_tensor.shape
        output_tensor = T.nnet.sigmoid(input_tensor.unwrap())
        return Tensor(output_tensor, shape=input_shape, name='output')


class Tanh(base_layer.BaseTanh):
    """Implement Tanh layer in Theano.

    See :any:`BaseTanh` for detail.
    """
    def _build(self, input_tensor):
        input_shape = input_tensor.shape
        output_tensor = T.tanh(input_tensor.unwrap())
        return Tensor(output_tensor, shape=input_shape, name='output')


class Softmax(base_layer.BaseSoftmax):
    """Implement Softmax layer in Theano.

    See :any:`BaseSoftmax` for detail.
    """
    def _build(self, input_tensor):
        input_shape = input_tensor.shape
        output_tensor = T.nnet.softmax(input_tensor.unwrap())
        return Tensor(output_tensor, shape=input_shape, name='output')
