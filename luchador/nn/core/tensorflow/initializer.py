"""Implement Initializer module in Tensorflow backend

See :any:`luchador.nn.core.base.initializer` for the interface.
"""
from __future__ import absolute_import

import tensorflow as tf
from tensorflow.contrib import layers as tf_layers

import luchador
from ..base import initializer as base_initializer


class InitializerMixin(object):
    """Provide TF-specific Initializer methods"""
    def _unwrap(self):
        """Returns the underlying TF native initializer"""
        return self._initializer

    def _sample(self):
        """Dummy sampling override
        In TF backend, initialization is handled by TF native initializers.
        """
        pass

    def _get_dtype(self):
        return tf.as_dtype(self.args['dtype'] or luchador.get_nn_dtype())


class Constant(InitializerMixin, base_initializer.BaseConstant):
    """Implement Constant in Tensorflow backend.

    See :any:`BaseConstant` for detail.
    """
    def _run_backend_specific_init(self):
        self._initializer = tf.constant_initializer(
            value=self.args['value'], dtype=self._get_dtype())


class Uniform(InitializerMixin, base_initializer.BaseUniform):
    """Implement Uniform in Tensorflow backend.

    See :any:`BaseUniform` for detail.
    """
    def _run_backend_specific_init(self):
        self._initializer = tf.random_uniform_initializer(
            minval=self.args['minval'], maxval=self.args['maxval'],
            seed=self.args['seed'], dtype=self._get_dtype())


class Normal(InitializerMixin, base_initializer.BaseNormal):
    """Implement Normal in Tensorflow backend.

    See :any:`BaseNormal` for detail.
    """
    def _run_backend_specific_init(self):
        self._initializer = tf.random_normal_initializer(
            mean=self.args['mean'], stddev=self.args['stddev'],
            seed=self.args['seed'], dtype=self._get_dtype())


class Xavier(InitializerMixin, base_initializer.BaseXavier):
    """Implement Xavier in Tensorflow backend.

    See :any:`BaseXavier` for detail.
    """
    def _run_backend_specific_init(self):
        self._initializer = tf_layers.xavier_initializer(
            uniform=self.args['uniform'], seed=self.args['seed'],
            dtype=self._get_dtype())
