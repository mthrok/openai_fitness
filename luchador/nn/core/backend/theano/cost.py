"""Implement Cost classes in Theano"""
from __future__ import absolute_import

import numbers

import numpy as np
import theano
import theano.tensor as T

from . import wrapper

__all__ = ['SSE', 'SigmoidCrossEntropy', 'SoftmaxCrossEntropy']
# pylint: disable=too-few-public-methods, no-member


def _unwrap_and_disconnect(tensor):
    if isinstance(tensor, numbers.Number):
        return tensor
    if isinstance(tensor, np.ndarray):
        return tensor
    return theano.gradient.disconnected_grad(tensor.unwrap())


def _mean_sum(x):
    return x.mean(axis=0).sum()


class SSE(object):
    """Implement SSE in Theano.

    See :any:`BaseSSE` for detail.
    """
    def _build(self, target, prediction):
        pred_ = prediction.unwrap()
        target_ = _unwrap_and_disconnect(target)
        error = T.square(target_ - pred_)
        output = error if self.args['elementwise'] else _mean_sum(error)
        shape = prediction.shape if self.args['elementwise'] else tuple()
        return wrapper.Tensor(output, shape=shape, name='output')


class SigmoidCrossEntropy(object):
    """Implement SigmoidCrossEntropy in Theano.

    See :any:`BaseSigmoidCrossEntropy` for detail.
    """
    def _build(self, target, logit):
        x = logit.unwrap()
        z = _unwrap_and_disconnect(target)
        ce = T.nnet.relu(x) - x * z + T.log(1 + T.exp(-abs(x)))

        output = ce if self.args['elementwise'] else _mean_sum(ce)
        shape = logit.shape if self.args['elementwise'] else tuple()
        return wrapper.Tensor(output, shape=shape, name='output')


class SoftmaxCrossEntropy(object):
    """Implement SoftmaxCrossEntropy in Theano.

    See :any:`BaseSoftmaxCrossEntropy` for detail.
    """
    def _build(self, target, logit):
        x = logit.unwrap()
        z = _unwrap_and_disconnect(target)

        xdev = x - x.max(1, keepdims=True)
        log_sm = xdev - T.log(T.sum(T.exp(xdev), axis=1, keepdims=True))
        ce = (- z * log_sm).sum(axis=1)

        output = ce if self.args['elementwise'] else _mean_sum(ce)
        shape = (logit.shape[0],) if self.args['elementwise'] else tuple()
        return wrapper.Tensor(output, shape=shape, name='output')
