"""Implement simple arithmetic operation layer classes in Theano"""
from __future__ import division
from __future__ import absolute_import

import theano.tensor as T

from ..wrapper import Tensor

__all__ = ['Add', 'Sub', 'TrueDiv']
# pylint: disable=no-member,too-few-public-methods,no-self-use,
# pylint: disable=attribute-defined-outside-init


class Add(object):
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


class Sub(object):
    """Implement Sub layer in Theano

    See :any: `BaseSub` for detail.
    """
    def _build(self, var_list):
        if len(var_list) != 2:
            raise ValueError('var_list must be 2 tensors')

        return var_list[0].__sub__(var_list[1], name='output')


class TrueDiv(object):
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
