"""Implement Activation Layers in Theano"""
from __future__ import division
from __future__ import absolute_import

import theano.tensor as T

from luchador.nn.core.base import fetch_initializer
from ..wrapper import Tensor, make_variable

__all__ = ['ReLU', 'LeakyReLU', 'Softplus', 'Sigmoid', 'Tanh', 'Softmax']
# pylint: disable=too-few-public-methods, no-self-use, no-member


class ReLU(object):
    """Implement ReLU layer in Theano.

    See :any:`ReLU` for detail.
    """
    def _build(self, input_tensor):
        """Build rectified linear activation operation on input tensor"""
        input_shape = input_tensor.shape
        output_tensor = T.nnet.relu(input_tensor.unwrap())
        return Tensor(output_tensor, shape=input_shape, name='output')


class LeakyReLU(object):
    """Implement LeakyReLU layer in Theano.

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
        alpha = self._get_alpha() if self.args['train'] else self.args['alpha']
        input_shape = input_tensor.shape
        output_tensor = T.nnet.relu(input_tensor.unwrap(), alpha=alpha)
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
