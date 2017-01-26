"""Implement Cost classes in Tensorflow"""

from __future__ import absolute_import

import tensorflow as tf

from luchador.nn.base import cost as base_cost
from . import wrapper

__all__ = ['SSE2', 'SigmoidCrossEntropy']


def _mean_sum(err):
    return tf.reduce_sum(tf.reduce_mean(err, reduction_indices=0))


class SSE2(base_cost.BaseSSE2):
    """Implement SSE2 in Tensorflow.

    See :any:`BaseSSE2` for detail.
    """
    def _build(self, target, prediction):
        with tf.name_scope('SSE'):
            pred_ = prediction.unwrap()
            target_ = tf.stop_gradient(target.unwrap())

            delta = tf.sub(target_, pred_, 'delta')
            if self.args.get('min_delta'):
                delta = tf.clip_by_value(
                    delta, self.args['min_delta'], self.args['max_delta'])
            err = tf.square(delta) / 2
            output = err if self.args['elementwise'] else _mean_sum(err)
            return wrapper.Tensor(output)


class SigmoidCrossEntropy(base_cost.BaseSigmoidCrossEntropy):
    """Implement SigmoidCrossEntropy in Tensorflow.

    See :any:`BaseSigmoidCrossEntropy` for detail.
    """
    def _build(self, target, logit):
        with tf.name_scope(self.__class__.__name__):
            sce = tf.nn.sigmoid_cross_entropy_with_logits(
                logit.unwrap(), tf.stop_gradient(target.unwrap()))

            output = sce if self.args['elementwise'] else _mean_sum(sce)
            return wrapper.Tensor(output)