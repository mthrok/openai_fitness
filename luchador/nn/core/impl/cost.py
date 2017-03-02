"""Implement Cost modules"""
from __future__ import absolute_import

from ..base.cost import BaseCost
from ..backend import cost

__all__ = ['SSE', 'SigmoidCrossEntropy']
# pylint: disable=too-few-public-methods, no-member


class SSE(cost.SSE, BaseCost):
    """Compute Sum-Squared-Error for the given target and prediction

    .. math::
        loss = (target - prediction) ^ {2}

    Parameters
    ----------
    elementwise : Bool
        When True, the cost tesnor returned by `build` method has the same
        shape as its input Tensors. When False, the cost tensor is reduced to
        scalar shape by taking average over batch and sum over feature.
        Defalut: False.
    """
    def __init__(self, elementwise=False):
        super(SSE, self).__init__(elementwise=elementwise)


class SigmoidCrossEntropy(cost.SigmoidCrossEntropy, BaseCost):
    """Directory computes classification entropy from logit

    .. math::
        loss = \\frac{-1}{n} \\sum\\limits_{n=1}^N \\left[ p_n \\log
                \\hat{p}_n + (1 - p_n) \\log(1 - \\hat{p}_n) \\right]

    Parameters
    ----------
    elementwise : Bool
        When True, the cost tesnor returned by `build` method has the same
        shape as its input Tensors. When False, the cost tensor is reduced to
        scalar shape by taking average over batch and sum over feature.
        Defalut: False.
    """
    def __init__(self, elementwise=False):
        super(SigmoidCrossEntropy, self).__init__(elementwise=elementwise)
