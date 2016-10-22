from __future__ import absolute_import

import tensorflow as tf

from ..base import (
    get_cost,
    BaseCost,
    BaseSSE2,
)
from .wrapper import Tensor

__all__ = [
    'BaseCost', 'get_cost',
    'SSE2', 'SigmoidCrossEntropy',
]


def mean_sum(err):
    err = tf.reduce_sum(err, reduction_indices=1)
    return tf.reduce_mean(err)


def _clipped_delta(target, prediction, min_delta, max_delta):
    delta = tf.sub(target, prediction, 'delta')
    if min_delta and max_delta:
        delta = tf.clip_by_value(delta, min_delta, max_delta)
    return delta


class SSE2(BaseSSE2):
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
            pred_ = prediction.unwrap()
            target_ = tf.stop_gradient(target.unwrap())
            delta = _clipped_delta(target_, pred_, min_delta, max_delta)
            err = tf.square(delta) / 2

            if self.args['elementwise']:
                output = Tensor(err)
            else:
                output = Tensor(mean_sum(err))
            return output


class SigmoidCrossEntropy(BaseCost):
    """Apply sigmoid activation followed by cross entropy """
    def build(self, target, logit):
        with tf.name_scope(self.__class__.__name__):
            ce = tf.nn.sigmoid_cross_entropy_with_logits(
                logit.unwrap(), tf.stop_gradient(target.unwrap()))

            if self.args['elementwise']:
                output = Tensor(ce)
            else:
                output = Tensor(mean_sum(ce))
            return output
