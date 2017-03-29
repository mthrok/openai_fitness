"""Implement Cost modules"""
from __future__ import absolute_import

import logging

from ..base.scope import variable_scope
from ..base.cost import BaseCost
from ..backend import cost
from ..backend import ops

__all__ = [
    'SSE', 'SigmoidCrossEntropy', 'SoftmaxCrossEntropy', 'NormalKLDivergence'
]
_LG = logging.getLogger(__name__)
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
    def __init__(self, elementwise=False, name='SSE'):
        super(SSE, self).__init__(elementwise=elementwise, name=name)


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
    def __init__(self, elementwise=False, name='SigmoidCrossEntropy'):
        super(SigmoidCrossEntropy, self).__init__(
            elementwise=elementwise, name=name)


class SoftmaxCrossEntropy(cost.SoftmaxCrossEntropy, BaseCost):
    """Directly computes classification entropy from logit

    Parameters
    ----------
    elementwise : Bool
        When True, the cost tesnor returned by `build` method has the same
        shape as its input Tensors. When False, the cost tensor is reduced to
        scalar shape by taking average over batch and sum over feature.
        Defalut: False.
    """
    def __init__(self, elementwise=False, name='SoftmaxCrossEntropy'):
        super(SoftmaxCrossEntropy, self).__init__(
            elementwise=elementwise, name=name)


class NormalKLDivergence(BaseCost):
    """KL-Divegence against univariate normal distribution.

    .. math::
        loss = \\frac{1}{2} (\\mu^2 + \\sigma^2 - \\log(clip(\\sigma^2)) -1)

    Parameters
    ----------
    elementwise : Bool
        When True, the cost tesnor returned by `build` method has the same
        shape as its input Tensors. When False, the cost tensor is reduced to
        scalar shape by taking average over batch and sum over feature.
        Defalut: False.

    clip_max, clip_min : float
        For numerical stability, :math:`\\sigma^2` is clipped before fed to
        :math:`\\log`.

    Notes
    -----
    The defalut values of clipping are chosen so as to prevent NaN and Inf
    which can happend as a direct result of ``log`` function.
    It can still generate huge gradients because the gradient of ``log``
    involves inverse of the ``stddev``. This might cause NaN at somewhere
    else in optimization process. When this occurs, you may want to narrow
    down clipping range or apply clipping to gradient separately.

    Reference; (ignoring ``mean``)

    +----------+----------+-----------+
    |  stddev  |     cost |  gradient |
    +==========+==========+===========+
    |        0 |      inf |       nan |
    +----------+----------+-----------+
    | 1.00e-05 | 1.10e+01 | -1.00e+05 |
    +----------+----------+-----------+
    | 1.00e-04 | 8.71e+00 | -1.00e+04 |
    +----------+----------+-----------+
    | 1.00e-03 | 6.41e+00 | -1.00e+03 |
    +----------+----------+-----------+
    | 1.00e-02 | 4.11e+00 | -1.00e+02 |
    +----------+----------+-----------+
    | 1.00e-01 | 1.81e+00 | -9.90e+00 |
    +----------+----------+-----------+
    | 1.00e+00 | 0.00e+00 |  0.00e+00 |
    +----------+----------+-----------+
    | 1.00e+01 | 4.72e+01 |  9.90e+00 |
    +----------+----------+-----------+
    | 1.00e+02 | 4.99e+03 |  1.00e+02 |
    +----------+----------+-----------+
    | 1.00e+03 | 5.00e+05 |  1.00e+03 |
    +----------+----------+-----------+
    | 1.00e+04 | 5.00e+07 |  1.00e+04 |
    +----------+----------+-----------+
    | 1.00e+05 | 5.00e+09 |  1.00e+05 |
    +----------+----------+-----------+
    """
    def __init__(
            self, elementwise=False, clip_max=1e+10, clip_min=1e-10,
            name='NormalKLDivergence'):
        super(NormalKLDivergence, self).__init__(
            elementwise=elementwise, clip_max=clip_max, clip_min=clip_min,
            name=name)

    def __call__(self, mean, stddev):
        return self.build(mean, stddev)

    def build(self, mean, stddev):
        """Build cost between target and prediction

        Parameters
        ----------
        mean, stddev : Tensor
            Tensor holding mean :math:`\\mu` and stddev :math:`\\sigma`

        Returns
        -------
        Tensor
            Tensor holding the cost between the given input tensors.
        """
        _LG.info(
            '  Building cost %s with mean: %s and stddev: %s',
            type(self).__name__, mean, stddev
        )
        self.input = {'mean': mean, 'stddev': stddev}
        with variable_scope(self.args['name']):
            self.output = self._build(mean, stddev)
            return self.output

    def _build(self, mean, stddev):
        min_, max_ = self.args['clip_min'], self.args['clip_max']
        mean2, stddev2 = [ops.square(val) for val in [mean, stddev]]
        clipped = ops.clip_by_value(stddev2, min_value=min_, max_value=max_)
        kl = 0.5 * (mean2 + stddev2 - ops.log(clipped) - 1)
        if self.args['elementwise']:
            return kl
        return ops.reduce_sum(ops.reduce_mean(kl, axis=0))
