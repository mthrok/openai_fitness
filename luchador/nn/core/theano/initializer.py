"""Implement Initializer module in Theano backend

See :any:`luchador.nn.core.base.initializer` for the interface.
"""
from __future__ import division
from __future__ import absolute_import

import numpy as np
from numpy.random import RandomState

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
                'Xavier initializer expects the shape to have 2 elements, '
                'e.g. [fan_in, fan_out]. Found: {}'.format(shape)
            )

        fan_in, fan_out = shape
        param = self._compute_param(fan_in, fan_out)
        return self._sample_value(shape, param)

    def _compute_param(self, fan_in, fan_out):
        if self.args['uniform']:
            x = np.sqrt(6. / (fan_in + fan_out))
            return {'low': -x, 'high': x}
        else:
            scale = np.sqrt(3. / (fan_in + fan_out))
            return {'loc': 0., 'scale': scale}

    def _sample_value(self, shape, param):
        if self.args['uniform']:
            values = self._rng.uniform(
                low=param['low'], high=param['high'], size=shape)
        else:
            values = self._rng.normal(
                loc=param['loc'], scale=param['scale'], size=shape)
        dtype = self.args['dtype'] or config.floatX
        return values.astype(dtype)


class XavierConv2D(Xavier):
    """Implement XavierConv2D in Theano backend.

    See :any:`BaseXavierConv2D` for detail.
    """
    def _sample(self, shape):
        # theano's filter shape is
        # (output_channels, input_channels, filter_rows, filter_columns)
        fan_in = shape[1] * shape[2] * shape[3]
        fan_out = shape[0] * shape[2] * shape[3]
        param = self._compute_param(fan_in, fan_out)
        return self._sample_value(shape, param)
