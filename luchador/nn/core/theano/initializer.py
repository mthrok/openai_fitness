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


def _sample_uniform(stddev, shape, rng):
    """Sample from uniform distribution in the way that
    resulting values have the given stddev"""
    bound = np.sqrt(3.0) * stddev
    return rng.uniform(low=-bound, high=bound, size=shape)


def _sample_truncated_normal(stddev, shape, rng):
    """Sample from truncated normal distribution in the way that
    resulting values have the given stddev"""
    scale = np.sqrt(1.3) * stddev
    return tnorm.rvs(-2, 2, scale=scale, size=shape, random_state=rng)


class Xavier(InitializerMixin, base_initializer.BaseXavier):
    """Implement Xavier in Theano backend.

    See :any:`BaseXavier` for detail.
    """
    def _sample(self, shape):
        if len(shape) not in [2, 4]:
            raise ValueError(
                'Xavier initializer expects the shape to be 2D or 4D.'
            )
        fan_ave = 0.5 * (shape[0] + shape[1]) * np.prod(shape[2:4])
        stddev = 1. / np.sqrt(fan_ave)
        value = (_sample_uniform(stddev, shape, self._rng)
                 if self.args['uniform'] else
                 _sample_truncated_normal(stddev, shape, self._rng))
        return value.astype(self.args['dtype'] or config.floatX)
