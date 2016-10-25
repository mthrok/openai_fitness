from __future__ import division
from __future__ import absolute_import

import numpy as np
from numpy.random import RandomState

from theano import config

from ..base import initializer as base_initializer
from . import random

__all__ = [
    'BaseInitializer', 'get_initializer',
    'Constant', 'Normal', 'Uniform', 'Xavier', 'XavierConv2D',
]

get_initializer = base_initializer.get_initializer
BaseInitializer = base_initializer.BaseInitializer


class Constant(BaseInitializer):
    """Initialize variale with constant value

    Parameters
    ----------
    value : number
        Value to initialize Variable

    dtype : str or None
        Data type to sample. If None, default dtype is used.
    """
    def __init__(self, value, dtype=None):
        super(Constant, self).__init__(value=value, dtype=dtype)

    def sample(self, shape):
        dtype = self.args['dtype'] or config.floatX
        return self.args['value'] * np.ones(shape, dtype=dtype)


class Uniform(BaseInitializer):
    def __init__(self, minval=0.0, maxval=1.0, seed=None, dtype=None):
        super(Uniform, self).__init__(
            minval=minval, maxval=maxval, seed=seed, dtype=dtype)
        self._rng = RandomState(seed) if seed else random.get_rng()

    def sample(self, shape):
        low, high = self.args['minval'], self.args['maxval']
        dtype = self.args['dtype'] or config.floatX
        values = self._rng.uniform(low=low, high=high, size=shape)
        return values.astype(dtype)


class Normal(BaseInitializer):
    def __init__(self, mean=0.0, stddev=1.0, seed=None, dtype=None):
        super(Normal, self).__init__(
            mean=mean, stddev=stddev, seed=seed, dtype=dtype)
        self._rng = RandomState(seed) if seed else random.get_rng()

    def sample(self, shape):
        loc, scale = self.args['mean'], self.args['stddev']
        dtype = self.args['dtype'] or config.floatX
        values = self._rng.normal(loc=loc, scale=scale, size=shape)
        return values.astype(dtype)


class Xavier(BaseInitializer):
    """Adoptation of xavier_initializer from tensorflow"""
    def __init__(self, uniform=True, seed=None, dtype=None):
        super(Xavier, self).__init__(uniform=uniform, seed=seed, dtype=dtype)
        self._rng = RandomState(seed) if seed else random.get_rng()

    def _compute_param(self, fan_in, fan_out):
        if self.args['uniform']:
            x = np.sqrt(6. / (fan_in + fan_out))
            return {'low': -x, 'high': x}
        else:
            scale = np.sqrt(3. / (fan_in + fan_out))
            return {'loc': 0., 'scale': scale}

    def _sample(self, shape, param):
        if self.args['uniform']:
            values = self._rng.uniform(
                low=param['low'], high=param['high'], size=shape)
        else:
            values = self._rng.normal(
                loc=param['loc'], scale=param['scale'], size=shape)
        dtype = self.args['dtype'] or config.floatX
        return values.astype(dtype)

    def sample(self, shape):
        if not len(shape) == 2:
            raise ValueError(
                'Xavier initializer expects the shape to have 2 elements, '
                'e.g. [fan_in, fan_out]. Found: {}'.format(shape)
            )

        fan_in, fan_out = shape[:2]
        param = self._compute_param(fan_in, fan_out)
        return self._sample(shape, param)


class XavierConv2D(Xavier):
    """Adoptation of xavier_initializer_conv2d from tensorflow"""
    def __init__(self, uniform=True, seed=None, dtype=None):
        super(XavierConv2D, self).__init__(uniform, seed, dtype)

    def sample(self, shape):
        # theano's filter shape is
        # (output_channels, input_channels, filter_rows, filter_columns)
        fan_in = shape[1] * shape[2] * shape[3]
        fan_out = shape[0] * shape[2] * shape[3]
        param = self._compute_param(fan_in, fan_out)
        return self._sample(shape, param)
