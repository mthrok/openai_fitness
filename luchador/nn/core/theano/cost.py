from __future__ import absolute_import

import logging

import theano
import theano.tensor as T

from ..base import SSE as BaseSSE
from .wrapper import Tensor

_LG = logging.getLogger(__name__)

__all__ = ['SSE2']


class SSE2(BaseSSE):
    """Compute Sum-Squared Error / 2
    TODO: Add math expression
    """
    def _validate_args(self, args):
        if (
                ('min_delta' in args and 'max_delta' in args) or
                ('min_delta' not in args and 'max_delta' not in args)
        ):
            return
        raise ValueError('When clipping delta, both '
                         '`min_delta` and `max_delta` must be provided')

    def build(self, target, prediction):
        target, prediction = target.unwrap(), prediction.unwrap()
        delta = theano.gradient.disconnected_grad(target) - prediction
        if self.args['min_delta']:
            delta = delta.clip(self.args['min_delta'], self.args['max_delta'])
        delta = T.square(delta) / 2
        err = delta.sum(axis=1).mean()
        output_shape = [1]
        return Tensor(err, output_shape)
