from __future__ import absolute_import

import tensorflow as tf
from tensorflow import (
    constant_initializer,
    random_normal_initializer,
    random_uniform_initializer,
)
from tensorflow.contrib.layers import (
    xavier_initializer,
    xavier_initializer_conv2d,
)

from luchador import get_nn_dtype
from ..base import (
    get_initializer,
    Initializer as BaseInitializer
)

__all__ = [
    'BaseInitializer', 'get_initializer',
    'Constant', 'Normal', 'Uniform', 'Xavier', 'XavierConv2D'
]


class TFInitializer(BaseInitializer):
    def unwrap(self):
        return self._initializer


class Constant(TFInitializer):
    def __init__(self, value, dtype=None):
        super(Constant, self).__init__(value=value, dtype=dtype)
        dtype = tf.as_dtype(dtype or get_nn_dtype())
        self._initializer = constant_initializer(value=value, dtype=dtype)


class Normal(TFInitializer):
    def __init__(self, mean=0.0, stddev=1.0, seed=None, dtype=None):
        super(Normal, self).__init__(
            mean=mean, stddev=stddev, seed=seed, dtype=dtype)
        dtype = tf.as_dtype(dtype or get_nn_dtype())
        self._initializer = random_normal_initializer(
            mean=mean, stddev=stddev, seed=seed, dtype=dtype)


class Uniform(TFInitializer):
    def __init__(self, minval=0.0, maxval=1.0, seed=None, dtype=None):
        super(Uniform, self).__init__(
            minval=minval, maxval=maxval, seed=seed, dtype=dtype)
        dtype = tf.as_dtype(dtype or get_nn_dtype())
        self._initializer = random_uniform_initializer(
            minval=minval, maxval=maxval, seed=seed, dtype=dtype)


class Xavier(TFInitializer):
    def __init__(self, uniform=True, seed=None, dtype=None):
        super(Xavier, self).__init__(
            uniform=uniform, seed=seed, dtype=dtype)
        dtype = tf.as_dtype(dtype or get_nn_dtype())
        self._initializer = xavier_initializer(
            uniform=uniform, seed=seed, dtype=dtype)


class XavierConv2D(TFInitializer):
    def __init__(self, uniform=True, seed=None, dtype=None):
        super(XavierConv2D, self).__init__(
            uniform=uniform, seed=seed, dtype=dtype)
        dtype = tf.as_dtype(dtype or get_nn_dtype())
        self._initializer = xavier_initializer_conv2d(
            uniform=uniform, seed=seed, dtype=dtype)
