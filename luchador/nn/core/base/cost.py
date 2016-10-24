"""Define common interface for Cost classes"""

from __future__ import absolute_import

import logging

from luchador.common import get_subclasses, StoreMixin

__all__ = [
    'BaseCost', 'get_cost',
    'BaseSSE2', 'BaseSigmoidCrossEntropy',
]

_LG = logging.getLogger(__name__)


class BaseCost(StoreMixin, object):
    """Common interface for cost computation

    Actual Cost class must implement `build` method.
    """
    def __init__(self, **args):
        """Validate args and set it as instance property

        See Also
        --------
        luchador.common.StoreMixin
            Underlying mechanism to store constructor arguments
        """
        super(BaseCost, self).__init__()
        self._store_args(**args)

    def __call__(self, target, prediction):
        """Convenience method to call `build`"""
        return self.build(target, prediction)

    def build(self, target, prediction):
        """Build cost between target and prediction

        Parameters
        ----------
        target : Tensor
            Tensor holding value for correct prediction

        prediction : Tensor
            Tensor holding the current prediction value.

        Returns
        -------
        Tensor
            Tensor holding the cost between the given input tensors.
        """
        return self._build(target, prediction)

    def _build(self, target, prediction):
        raise NotImplementedError(
            '`_build` method is not implemented for {}.{}.'
            .format(type(self).__module__, type(self).__name__)
        )


def get_cost(name):
    for class_ in get_subclasses(BaseCost):
        if class_.__name__ == name:
            return class_
    raise ValueError('Unknown Cost: {}'.format(name))


###############################################################################
class BaseSSE2(BaseCost):
    """Compute Sum-Squared-Error / 2.0 for the given target and prediction

    TODO: Add math expression

    Parameters
    ----------
    max_delta, min_delta : float
        Clip the difference between target value and prediction values

    elementwise : Bool
        When true, the cost tesnor returned by `build` method has the same
        shape as its input Tensors. When False, the cost tensor is flattened
        to scalar shape by taking average over batch and sum over feature.
        Default: False
    """
    def __init__(self, max_delta=None, min_delta=None, elementwise=False):
        super(BaseSSE2, self).__init__(
            max_delta=max_delta, min_delta=min_delta, elementwise=elementwise)


class BaseSigmoidCrossEntropy(BaseCost):
    """Directory computes classification entropy from logit

    Parameters
    ----------
    elementwise : Bool
        When true, the cost tesnor returned by `build` method has the same
        shape as its input Tensors. When False, the cost tensor is flattened
        to scalar shape by taking average over batch and sum over feature.
        Defalut False
    """
    def __init__(self, elementwise=False):
        super(BaseSigmoidCrossEntropy, self).__init__(elementwise=elementwise)
