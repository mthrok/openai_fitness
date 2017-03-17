"""Implement Layer classes in Tensorflow"""
from __future__ import division
from __future__ import absolute_import

import logging

import tensorflow as tf

import luchador
from ..wrapper import Tensor

__all__ = ['TrueDiv']
_LG = logging.getLogger(__name__)
# pylint:disable=no-member,no-self-use,attribute-defined-outside-init


class TrueDiv(object):
    """Implement TrueDiv in Tensorflow.

    See :any:`BaseTrueDiv` for detail.
    """
    def _instantiate_denominator(self, dtype):
        self._denom = tf.constant(
            self.args['denom'], dtype=dtype, name='denominator')

    def _build(self, input_tensor):
        dtype = input_tensor.dtype
        tensor = input_tensor.unwrap()
        if 'int' in input_tensor.dtype:
            dtype = luchador.get_nn_dtype()
            tensor = tf.cast(tensor, dtype)

        if self._denom is None:
            self._instantiate_denominator(dtype)

        output = tf.truediv(tensor, self._denom, 'ouptut')
        return Tensor(output, name='output')
