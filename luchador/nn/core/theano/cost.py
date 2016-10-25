"""Implement Cost classes in Theano"""

from __future__ import absolute_import

import logging

import theano
import theano.tensor as T

from ..base import cost as base_cost
from . import wrapper

__all__ = [
    'BaseCost', 'get_cost',
    'SSE2', 'SigmoidCrossEntropy',
]

_LG = logging.getLogger(__name__)

get_cost = base_cost.get_cost
BaseCost = base_cost.BaseCost


def _mean_sum(x):
    return x.sum(axis=1).mean()


class SSE2(base_cost.BaseSSE2):
    """Implement SSE2 in Theano"""
    def validate_args(self, min_delta, max_delta, **kwargs):
        if (min_delta and max_delta) or (not max_delta and not min_delta):
            return
        raise ValueError('When clipping delta, both '
                         '`min_delta` and `max_delta` must be provided')

    def _build(self, target, prediction):
        target_, pred_ = target.unwrap(), prediction.unwrap()
        delta = theano.gradient.disconnected_grad(target_) - pred_
        if self.args['min_delta']:
            delta = delta.clip(self.args['min_delta'], self.args['max_delta'])
        delta = T.square(delta) / 2

        output = delta if self.args['elementwise'] else _mean_sum(delta)
        shape = target.shape if self.args['elementwise'] else (1,)
        return wrapper.Tensor(output, shape=shape)


class SigmoidCrossEntropy(base_cost.BaseSigmoidCrossEntropy):
    """Implement SCE in Theano"""
    def _build(self, target, logit):
        x = logit.unwrap()
        z = theano.gradient.disconnected_grad(target.unwrap())
        ce = T.nnet.relu(x) - x * z + T.log(1 + T.exp(-abs(x)))

        output = ce if self.args['elementwise'] else _mean_sum(ce)
        shape = target.shape if self.args['elementwise'] else (1,)
        return wrapper.Tensor(output, shape=shape)
