"""Implement Activation Layers in Theano"""
from __future__ import division
from __future__ import absolute_import

import theano.tensor as T

from ..wrapper import Tensor

__all__ = ['ReLU', 'Softplus', 'Sigmoid', 'Tanh', 'Softmax']
# pylint: disable=too-few-public-methods, no-self-use


class ReLU(object):
    """Implement ReLU layer in Theano.

    See :any:`ReLU` for detail.
    """
    def _build(self, input_tensor):
        """Build rectified linear activation operation on input tensor"""
        input_shape = input_tensor.shape
        output_tensor = T.nnet.relu(input_tensor.unwrap())
        return Tensor(output_tensor, shape=input_shape, name='output')


class Softplus(object):
    """Implemente Softplus layer in Theano.

    See :any:`Softplus` for detail.
    """
    def _build(self, input_tensor):
        input_shape = input_tensor.shape
        output_tensor = T.nnet.softplus(input_tensor.unwrap())
        return Tensor(output_tensor, shape=input_shape, name='output')


class Sigmoid(object):
    """Implement Sigmoid layer in Theano.

    See :any:`Sigmoid` for detail.
    """
    def _build(self, input_tensor):
        input_shape = input_tensor.shape
        output_tensor = T.nnet.sigmoid(input_tensor.unwrap())
        return Tensor(output_tensor, shape=input_shape, name='output')


class Tanh(object):
    """Implement Tanh layer in Theano.

    See :any:`Tanh` for detail.
    """
    def _build(self, input_tensor):
        input_shape = input_tensor.shape
        output_tensor = T.tanh(input_tensor.unwrap())
        return Tensor(output_tensor, shape=input_shape, name='output')


class Softmax(object):
    """Implement Softmax layer in Theano.

    See :any:`Softmax` for detail.
    """
    def _build(self, input_tensor):
        input_shape = input_tensor.shape
        output_tensor = T.nnet.softmax(input_tensor.unwrap())
        return Tensor(output_tensor, shape=input_shape, name='output')
