from __future__ import absolute_import

import tensorflow as tf

from ..base import (
    SSE as BaseSSE,
)
from .wrapper import Tensor

__all__ = ['SSE2']


def _clipped_delta(target, prediction, min_delta, max_delta):
    delta = tf.sub(target, prediction, 'delta')
    if min_delta and max_delta:
        delta = tf.clip_by_value(delta, min_delta, max_delta)
    return delta


class SSE2(BaseSSE):
    """Compute Sum-Squared-Error / 2.0 for the given target and prediction"""
    def _validate_args(self, args):
        if (
                ('min_delta' in args and 'max_delta' in args) or
                ('min_delta' not in args and 'max_delta' not in args)
        ):
            return
        raise ValueError('When clipping delta, both '
                         '`min_delta` and `max_delta` must be provided')

    def build(self, target, prediction):
        """Build error tensor for target and prediction.
        TODO: Add Math expression here.
        """
        with tf.name_scope('SSE'):
            min_delta = self.args.get('min_delta')
            max_delta = self.args.get('max_delta')
            target = tf.stop_gradient(target.unwrap())
            delta = _clipped_delta(
                target, prediction.unwrap(), min_delta, max_delta)
            err = tf.square(delta)
            err = tf.reduce_sum(err/2, reduction_indices=1, name='SSE2')
            # TODO: Remove this?
            err = tf.reduce_mean(err, name='SSE_over_batch')
            return Tensor(err)
