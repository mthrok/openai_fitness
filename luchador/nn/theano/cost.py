"""Implement Cost classes in Theano"""
from __future__ import absolute_import

import logging

import theano
import theano.tensor as T

from luchador.nn.base import cost as base_cost
from . import wrapper

__all__ = ['SSE', 'SigmoidCrossEntropy']

_LG = logging.getLogger(__name__)


def _mean_sum(x):
    return x.mean(axis=0).sum()


class SSE(base_cost.BaseSSE):
    """Implement SSE in Theano.

    See :any:`BaseSSE` for detail.
    """
    def _build(self, target, prediction):
        pred_ = prediction.unwrap()
        target_ = theano.gradient.disconnected_grad(target.unwrap())
        error = T.square(target_ - pred_)
        output = error if self.args['elementwise'] else _mean_sum(error)
        shape = target.shape if self.args['elementwise'] else (1,)
        return wrapper.Tensor(output, shape=shape)


class SigmoidCrossEntropy(base_cost.BaseSigmoidCrossEntropy):
    """Implement SigmoidCrossEntropy in Theano.

    See :any:`BaseSigmoidCrossEntropy` for detail.
    """
    def _build(self, target, logit):
        x = logit.unwrap()
        z = theano.gradient.disconnected_grad(target.unwrap())
        ce = T.nnet.relu(x) - x * z + T.log(1 + T.exp(-abs(x)))

        output = ce if self.args['elementwise'] else _mean_sum(ce)
        shape = target.shape if self.args['elementwise'] else (1,)
        return wrapper.Tensor(output, shape=shape)
