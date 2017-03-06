"""Implement simple arithmetic operation layer classes in Theano"""
from __future__ import division
from __future__ import absolute_import

import theano.tensor as T

from ..wrapper import Tensor

__all__ = ['TrueDiv']
# pylint: disable=no-member,too-few-public-methods,no-self-use,
# pylint: disable=attribute-defined-outside-init


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
