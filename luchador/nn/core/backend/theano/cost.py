"""Implement Cost classes in Theano"""
from __future__ import absolute_import

import theano
import theano.tensor as T

from . import wrapper

__all__ = ['SSE', 'SigmoidCrossEntropy', 'SoftmaxCrossEntropy']
# pylint: disable=too-few-public-methods, no-member


def _mean_sum(x):
    return x.mean(axis=0).sum()


class SSE(object):
    """Implement SSE in Theano.

    See :any:`BaseSSE` for detail.
    """
    def _build(self, target, prediction):
        pred_ = prediction.unwrap()
        target_ = theano.gradient.disconnected_grad(target.unwrap())
        error = T.square(target_ - pred_)
        output = error if self.args['elementwise'] else _mean_sum(error)
        shape = target.shape if self.args['elementwise'] else tuple()
        return wrapper.Tensor(output, shape=shape, name='output')


class SigmoidCrossEntropy(object):
    """Implement SigmoidCrossEntropy in Theano.

    See :any:`BaseSigmoidCrossEntropy` for detail.
    """
    def _build(self, target, logit):
        x = logit.unwrap()
        z = theano.gradient.disconnected_grad(target.unwrap())
        ce = T.nnet.relu(x) - x * z + T.log(1 + T.exp(-abs(x)))

        output = ce if self.args['elementwise'] else _mean_sum(ce)
        shape = target.shape if self.args['elementwise'] else tuple()
        return wrapper.Tensor(output, shape=shape, name='output')


class SoftmaxCrossEntropy(object):
    """Implement SoftmaxCrossEntropy in Theano.

    See :any:`BaseSoftmaxCrossEntropy` for detail.
    """
    def _build(self, target, logit):
        x = logit.unwrap()
        z = theano.gradient.disconnected_grad(target.unwrap())

        xdev = x - x.max(1, keepdims=True)
        log_sm = xdev - T.log(T.sum(T.exp(xdev), axis=1, keepdims=True))
        ce = (- z * log_sm).sum(axis=1)

        output = ce if self.args['elementwise'] else _mean_sum(ce)
        shape = (target.shape[0],) if self.args['elementwise'] else tuple()
        return wrapper.Tensor(output, shape=shape, name='output')
