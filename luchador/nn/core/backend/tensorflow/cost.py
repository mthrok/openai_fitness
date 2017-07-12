"""Implement Cost classes in Tensorflow"""
from __future__ import absolute_import

import numbers

import numpy as np
import tensorflow as tf

from . import wrapper

__all__ = ['SSE', 'SigmoidCrossEntropy', 'SoftmaxCrossEntropy']
# pylint: disable=too-few-public-methods, no-member


def _get_tf_tensor(tensor, ref_tf_tensor):
    if isinstance(tensor, numbers.Number):
        return tensor * tf.ones_like(ref_tf_tensor)
    if isinstance(tensor, np.ndarray):
        return tf.constant(tensor, dtype=ref_tf_tensor.dtype)
    return tensor.unwrap()


def _mean_sum(err):
    return tf.reduce_sum(tf.reduce_mean(err, reduction_indices=0))


class SSE(object):
    """Implement SSE in Tensorflow.

    See :any:`BaseSSE` for detail.
    """
    def _build(self, target, prediction):
        pred_ = prediction.unwrap()
        target_ = tf.stop_gradient(_get_tf_tensor(target, pred_))
        err = tf.square(target_ - pred_)
        output = err if self.args['elementwise'] else _mean_sum(err)
        return wrapper.Tensor(output, name='output')


class SigmoidCrossEntropy(object):
    """Implement SigmoidCrossEntropy in Tensorflow.

    See :any:`BaseSigmoidCrossEntropy` for detail.
    """
    def _build(self, target, logit):
        logits = logit.unwrap()
        labels = tf.stop_gradient(_get_tf_tensor(target, logits))
        sce = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels, logits=logits)
        output = sce if self.args['elementwise'] else _mean_sum(sce)
        return wrapper.Tensor(output)


class SoftmaxCrossEntropy(object):
    """Implement SoftmaxCrossEntropy in Tensorflow.

    See :any:`BaseSoftmaxCrossEntropy` for detail.
    """
    def _build(self, target, logit):
        x = logit.unwrap()
        z = tf.stop_gradient(_get_tf_tensor(target, x))
        ce = tf.nn.softmax_cross_entropy_with_logits(labels=z, logits=x)
        output = ce if self.args['elementwise'] else _mean_sum(ce)
        return wrapper.Tensor(output, name='output')
