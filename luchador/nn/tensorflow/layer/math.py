"""Implement Layer classes in Tensorflow"""
from __future__ import division
from __future__ import absolute_import

import logging

import tensorflow as tf

import luchador
from ...base import layer as base_layer
from ..wrapper import Tensor
from .common import LayerMixin

__all__ = [
    'Add', 'Sub', 'TrueDiv', 'Mean', 'Sin', 'Cos',
]

_LG = logging.getLogger(__name__)


class Add(LayerMixin, base_layer.BaseAdd):
    """Implement Add layer in Tensorflow

    See :any: `BaseAdd` for detail.
    """
    def _build(self, var_list):
        if len(var_list) < 2:
            raise ValueError('var_list must contain at least 2 tensors')

        ret = var_list[0]
        for var in var_list[1:-1]:
            ret = ret + var
        return ret.__add__(var_list[-1], name='output')


class Sub(LayerMixin, base_layer.BaseAdd):
    """Implement Sub layer in Tensorflow

    See :any: `BaseSub` for detail.
    """
    def _build(self, var_list):
        if len(var_list) != 2:
            raise ValueError('var_list must be 2 tensors')

        return var_list[0].__sub__(var_list[1], name='output')


class TrueDiv(LayerMixin, base_layer.BaseTrueDiv):
    """Implement TrueDiv in Tensorflow.

    See :any:`BaseTrueDiv` for detail.
    """
    def _instantiate_denominator(self, dtype):
        self.denom = tf.constant(
            self.args['denom'], dtype=dtype, name='denominator')

    def _build(self, input_tensor):
        dtype = input_tensor.dtype
        tensor = input_tensor.unwrap()
        if 'int' in input_tensor.dtype:
            dtype = luchador.get_nn_dtype()
            tensor = tf.cast(tensor, dtype)

        if self.denom is None:
            self._instantiate_denominator(dtype)

        output = tf.truediv(tensor, self.denom, 'ouptut')
        return Tensor(output, name='output')


class Mean(LayerMixin, base_layer.BaseMean):
    """Implement Mean layer in Tensorflow.

    See :any:`BaseMean` for detail.
    """
    def _build(self, input_tensor):
        return input_tensor.mean(
            axis=self.args['axis'], keep_dims=self.args['keep_dims'],
            name='output')


class Sin(LayerMixin, base_layer.BaseSin):
    """Implement Sin in Tensorflow.

    See :any:`BaseSin` for detail.
    """
    def _build(self, input_tensor):
        output = tf.sin(input_tensor.unwrap(), 'output')
        return Tensor(output, name='output')


class Cos(LayerMixin, base_layer.BaseCos):
    """Implement Cos in Tensorflow.

    See :any:`BaseCos` for detail.
    """
    def _build(self, input_tensor):
        output = tf.cos(input_tensor.unwrap(), 'output')
        return Tensor(output, name='output')
