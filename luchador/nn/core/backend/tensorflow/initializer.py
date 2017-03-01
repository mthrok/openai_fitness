"""Implement Initializer module in Tensorflow backend

See :py:mod:`luchador.nn.core.base.initializer` for the interface.
"""
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf
from tensorflow.contrib import layers as tf_layers

import luchador

__all__ = ['Constant', 'Uniform', 'Normal', 'Xavier', 'Kaiming']
# pylint: disable=too-few-public-methods
# pylint: disable=attribute-defined-outside-init,no-member


class InitializerMixin(object):
    """Provide TF-specific Initializer methods"""
    def unwrap(self):
        """Returns the underlying TF native initializer"""
        return self._initializer

    def _sample(self):
        """Dummy sampling override
        In TF backend, initialization is handled by TF native initializers.
        """
        pass

    def _get_dtype(self):
        return tf.as_dtype(self.args['dtype'] or luchador.get_nn_dtype())


class Constant(InitializerMixin):
    """Implement Constant in Tensorflow backend.

    See :any:`ConstantInitializer` for detail.
    """
    def _run_backend_specific_init(self):
        self._initializer = tf.constant_initializer(
            value=self.args['value'], dtype=self._get_dtype())


class Uniform(InitializerMixin):
    """Implement Uniform in Tensorflow backend.

    See :any:`UniformInitializer` for detail.
    """
    def _run_backend_specific_init(self):
        self._initializer = tf.random_uniform_initializer(
            minval=self.args['minval'], maxval=self.args['maxval'],
            seed=self.args['seed'], dtype=self._get_dtype())


class Normal(InitializerMixin):
    """Implement Normal in Tensorflow backend.

    See :any:`NormalInitializer` for detail.
    """
    def _run_backend_specific_init(self):
        self._initializer = tf.random_normal_initializer(
            mean=self.args['mean'], stddev=self.args['stddev'],
            seed=self.args['seed'], dtype=self._get_dtype())


class Xavier(InitializerMixin):
    """Implement Xavier in Tensorflow backend.

    See :any:`XavierInitializer` for detail.
    """
    def _run_backend_specific_init(self):
        self._initializer = tf_layers.xavier_initializer(
            uniform=self.args['uniform'], seed=self.args['seed'],
            dtype=self._get_dtype())


class Kaiming(InitializerMixin):
    """Implement Kaiming He initializer in Tensorflow backend.

    See :any:`KaimingInitializer` for detail.
    """
    def _run_backend_specific_init(self):
        self._initializer = tf_layers.variance_scaling_initializer(
            mode='FAN_IN', factor=1, uniform=self.args['uniform'],
            seed=self.args['seed'], dtype=self._get_dtype())
