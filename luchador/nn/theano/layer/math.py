"""Implement simple arithmetic operation layer classes in Theano"""
from __future__ import division
from __future__ import absolute_import

import theano.tensor as T

from ...base import layer as base_layer
from ..wrapper import Tensor

__all__ = [
    'Add', 'Sub', 'TrueDiv', 'Mean', 'Sin', 'Cos'
]


class Add(base_layer.BaseAdd):
    """Implement Add layer in Theano

    See :any: `BaseAdd` for detail.
    """
    def _build(self, var_list):
        if len(var_list) < 2:
            raise ValueError('var_list must contain at least 2 tensors')

        ret = var_list[0]
        for var in var_list[1:-1]:
            ret = ret + var
        return ret.__add__(var_list[-1], name='output')


class Sub(base_layer.BaseAdd):
    """Implement Sub layer in Theano

    See :any: `BaseSub` for detail.
    """
    def _build(self, var_list):
        if len(var_list) != 2:
            raise ValueError('var_list must be 2 tensors')

        return var_list[0].__sub__(var_list[1], name='output')


class TrueDiv(base_layer.BaseTrueDiv):
    """Implement TrueDiv layer in Theano.

    See :any:`BaseTrueDiv` for detail.
    """
    def _instantiate_denominator(self, dtype):
        self.denom = T.constant(
            self.args['denom'], dtype=dtype, name='denominator')

    def _build(self, input_tensor):
        if self.denom is None:
            self._instantiate_denominator(input_tensor.dtype)
        output = input_tensor.unwrap() / self.denom
        return Tensor(output, shape=input_tensor.shape, name='output')


class Mean(base_layer.BaseMean):
    """Implement Mean layer in Theano.

    See :any:`BaseMean` for detail.
    """
    def _build(self, input_tensor):
        return input_tensor.mean(name='output', **self.args)


class Sin(base_layer.BaseSin):
    """Implement Sin layer in Theano

    See :any:`BaseSin` for detail.
    """
    def _build(self, input_tensor):
        input_shape = input_tensor.shape
        output_tensor = T.sin(input_tensor.unwrap())
        return Tensor(output_tensor, shape=input_shape, name='output')


class Cos(base_layer.BaseCos):
    """Implement Cos layer in Theano

    See :any:`BaseSin` for detail.
    """
    def _build(self, input_tensor):
        input_shape = input_tensor.shape
        output_tensor = T.cos(input_tensor.unwrap())
        return Tensor(output_tensor, shape=input_shape, name='output')
