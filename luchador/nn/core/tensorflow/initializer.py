from __future__ import absolute_import

import tensorflow as tf
from tensorflow.contrib import layers as tf_layers

import luchador
from ..base import initializer as base_initializer

__all__ = [
    'TFInitializerMixin',
    'Constant', 'Normal', 'Uniform', 'Xavier', 'XavierConv2D'
]


class TFInitializerMixin(object):
    """Provide TF-specific Initializer methods"""
    def unwrap(self):
        """Returns the underlying TF native initializer"""
        return self._initializer

    def sample(self):
        """Dummy sampling override
        In TF backend, initialization is handled by TF native initializers.
        """
        pass


class Constant(TFInitializerMixin, base_initializer.BaseInitializer):
    def __init__(self, value, dtype=None):
        super(Constant, self).__init__(value=value, dtype=dtype)
        dtype = tf.as_dtype(dtype or luchador.get_nn_dtype())
        self._initializer = tf.constant_initializer(value=value, dtype=dtype)


class Normal(TFInitializerMixin, base_initializer.BaseInitializer):
    def __init__(self, mean=0.0, stddev=1.0, seed=None, dtype=None):
        super(Normal, self).__init__(
            mean=mean, stddev=stddev, seed=seed, dtype=dtype)
        dtype = tf.as_dtype(dtype or luchador.get_nn_dtype())
        self._initializer = tf.random_normal_initializer(
            mean=mean, stddev=stddev, seed=seed, dtype=dtype)


class Uniform(TFInitializerMixin, base_initializer.BaseInitializer):
    def __init__(self, minval=0.0, maxval=1.0, seed=None, dtype=None):
        super(Uniform, self).__init__(
            minval=minval, maxval=maxval, seed=seed, dtype=dtype)
        dtype = tf.as_dtype(dtype or luchador.get_nn_dtype())
        self._initializer = tf.random_uniform_initializer(
            minval=minval, maxval=maxval, seed=seed, dtype=dtype)


class Xavier(TFInitializerMixin, base_initializer.BaseInitializer):
    def __init__(self, uniform=True, seed=None, dtype=None):
        super(Xavier, self).__init__(
            uniform=uniform, seed=seed, dtype=dtype)
        dtype = tf.as_dtype(dtype or luchador.get_nn_dtype())
        self._initializer = tf_layers.xavier_initializer(
            uniform=uniform, seed=seed, dtype=dtype)


class XavierConv2D(TFInitializerMixin, base_initializer.BaseInitializer):
    def __init__(self, uniform=True, seed=None, dtype=None):
        super(XavierConv2D, self).__init__(
            uniform=uniform, seed=seed, dtype=dtype)
        dtype = tf.as_dtype(dtype or luchador.get_nn_dtype())
        self._initializer = tf_layers.xavier_initializer_conv2d(
            uniform=uniform, seed=seed, dtype=dtype)
