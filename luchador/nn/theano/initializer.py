"""Implement Initializer module in Theano backend

See :py:mod:`luchador.nn.core.base.initializer` for the interface.
"""
from __future__ import division
from __future__ import absolute_import

import abc

import numpy as np
from numpy.random import RandomState
from scipy.stats import truncnorm as tnorm

from theano import config as theano_config

from luchador.nn.base import initializer as base_initializer
from .random import get_rng

__all__ = [
    'InitializerMixin',
    'Constant', 'Uniform', 'Normal', 'Xavier', 'Kaiming'
]


class InitializerMixin(object):  # pylint: disable=too-few-public-methods
    """Provide Theano-specific Initializer methods"""
    def _run_backend_specific_init(self):
        if 'seed' in self.args:
            seed = self.args['seed']
            self._rng = RandomState(seed) if seed else get_rng()

    def _sample(self, shape):
        dtype = self.args['dtype'] or theano_config.floatX
        return self._sample_values(shape).astype(dtype)

    @abc.abstractmethod
    def _sample_values(self, shape):
        pass


class Constant(InitializerMixin, base_initializer.BaseConstant):
    """Implement Constant in Theano backend.

    See :any:`BaseConstant` for detail.
    """
    def _sample_values(self, shape):
        return self.args['value'] * np.ones(shape)


class Uniform(InitializerMixin, base_initializer.BaseUniform):
    """Implement Uniform in Theano backend.

    See :any:`BaseUniform` for detail.
    """
    def _sample_values(self, shape):
        low, high = self.args['minval'], self.args['maxval']
        return self._rng.uniform(low=low, high=high, size=shape)


class Normal(InitializerMixin, base_initializer.BaseNormal):
    """Implement Normal in Theano backend.

    See :any:`BaseNormal` for detail.
    """
    def _sample_values(self, shape):
        loc, scale = self.args['mean'], self.args['stddev']
        return self._rng.normal(loc=loc, scale=scale, size=shape)


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
    def _sample_values(self, shape):
        if len(shape) not in [2, 4]:
            raise ValueError(
                'Xavier initializer expects the shape to be 2D or 4D.'
            )
        fan_ave = 0.5 * (shape[0] + shape[1]) * np.prod(shape[2:4])
        stddev = 1. / np.sqrt(fan_ave)
        if self.args['uniform']:
            return _sample_uniform(stddev, shape, self._rng)
        else:
            return _sample_truncated_normal(stddev, shape, self._rng)


class Kaiming(InitializerMixin, base_initializer.BaseKaiming):
    """Implement Kaiming initialization in Theano backend.

    See :any:`BaseKaiming` for detail.
    """
    def _sample_values(self, shape):
        if len(shape) not in [2, 4]:
            raise ValueError(
                'Kaiming initializer expects the shape to be 2D or 4D.'
            )

        if len(shape) == 4:
            fan_in = np.prod(shape[1:])
        elif len(shape) == 2:
            fan_in = shape[0]

        stddev = 1. / np.sqrt(fan_in)
        if self.args['uniform']:
            return _sample_uniform(stddev, shape, self._rng)
        else:
            return _sample_truncated_normal(stddev, shape, self._rng)
