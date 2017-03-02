"""Implement Cost classes in Tensorflow"""
from __future__ import absolute_import

import tensorflow as tf

from . import wrapper

__all__ = ['SSE', 'SigmoidCrossEntropy']
# pylint: disable=too-few-public-methods, no-member


def _mean_sum(err):
    return tf.reduce_sum(tf.reduce_mean(err, reduction_indices=0))


class SSE(object):
    """Implement SSE in Tensorflow.

    See :any:`BaseSSE` for detail.
    """
    def _build(self, target, prediction):
        with tf.name_scope('SSE'):
            pred_ = prediction.unwrap()
            target_ = tf.stop_gradient(target.unwrap())
            err = tf.square(target_ - pred_)
            output = err if self.args['elementwise'] else _mean_sum(err)
            return wrapper.Tensor(output)


class SigmoidCrossEntropy(object):
    """Implement SigmoidCrossEntropy in Tensorflow.

    See :any:`BaseSigmoidCrossEntropy` for detail.
    """
    def _build(self, target, logit):
        with tf.name_scope(self.__class__.__name__):
            logits = logit.unwrap()
            labels = tf.stop_gradient(target.unwrap())
            sce = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=labels, logits=logits)

            output = sce if self.args['elementwise'] else _mean_sum(sce)
            return wrapper.Tensor(output)
