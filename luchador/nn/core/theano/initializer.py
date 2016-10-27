"""Implement Initializer module in Theano backend

See :any:`luchador.nn.core.base.initializer` for the interface.
"""
from __future__ import division
from __future__ import absolute_import

import numpy as np
from numpy.random import RandomState
from scipy.stats import truncnorm as tnorm

from theano import config

from ..base import initializer as base_initializer
from . import random


class InitializerMixin(object):
    """Provide Theano-specific Initializer methods"""
    def _run_backend_specific_init(self):
        if 'seed' in self.args:
            seed = self.args['seed']
            self._rng = RandomState(seed) if seed else random.get_rng()


class Constant(InitializerMixin, base_initializer.BaseConstant):
    """Implement Constant in Theano backend.

    See :any:`BaseConstant` for detail.
    """
    def _sample(self, shape):
        dtype = self.args['dtype'] or config.floatX
        return self.args['value'] * np.ones(shape, dtype=dtype)


class Uniform(InitializerMixin, base_initializer.BaseUniform):
    """Implement Uniform in Theano backend.

    See :any:`BaseUniform` for detail.
    """
    def _sample(self, shape):
        low, high = self.args['minval'], self.args['maxval']
        dtype = self.args['dtype'] or config.floatX
        values = self._rng.uniform(low=low, high=high, size=shape)
        return values.astype(dtype)


class Normal(InitializerMixin, base_initializer.BaseNormal):
    """Implement Normal in Theano backend.

    See :any:`BaseNormal` for detail.
    """
    def _sample(self, shape):
        loc, scale = self.args['mean'], self.args['stddev']
        dtype = self.args['dtype'] or config.floatX
        values = self._rng.normal(loc=loc, scale=scale, size=shape)
        return values.astype(dtype)


class Xavier(InitializerMixin, base_initializer.BaseXavier):
    """Implement Xavier in Theano backend.

    See :any:`BaseXavier` for detail.
    """
    def _sample(self, shape):
        if not len(shape) == 2:
            raise ValueError(
                'Xavier initializer expects the shape to have 2 elements.'
                'Found: {}'.format(shape)
            )

        fan_out, fan_in = shape[0], shape[1]
        scale = np.sqrt(6. / (fan_in + fan_out))
        if self.args['uniform']:
            value = self._sample_uniform(scale, shape)
        else:
            value = self._sample_truncated_normal(scale, shape)
        return value.astype(self.args['dtype'] or config.floatX)

    def _sample_uniform(self, scale, shape):
        return self._rng.uniform(low=-scale, high=scale, size=shape)

    def _sample_truncated_normal(self, scale, shape):
        return tnorm.rvs(
            -1, 1, scale=scale, size=shape, random_state=self._rng)


class XavierConv2D(Xavier):
    """Implement XavierConv2D in Theano backend.

    See :any:`BaseXavierConv2D` for detail.
    """
    def _sample(self, shape):
        if not len(shape) == 4:
            raise ValueError(
                'Xavier conv2d initializer expects the shape with 4 elements.'
                'Found: {}'.format(shape)
            )
        # theano's filter shape is
        # (output_channels, input_channels, filter_rows, filter_columns)
        fan_in = shape[1] * shape[2] * shape[3]
        fan_out = shape[0] * shape[2] * shape[3]
        scale = np.sqrt(6. / (fan_in + fan_out))
        if self.args['uniform']:
            value = self._sample_uniform(scale, shape)
        else:
            value = self._sample_truncated_normal(scale, shape)
        return value.astype(self.args['dtype'] or config.floatX)
