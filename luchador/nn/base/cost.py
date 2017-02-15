"""Define common interface for Cost classes"""
from __future__ import absolute_import

import abc
import logging

import luchador.util


_LG = logging.getLogger(__name__)


class BaseCost(luchador.util.StoreMixin, object):
    """Define common interface for cost computation class"""
    __metaclass__ = abc.ABCMeta

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
            Tensor holding the correct prediction value.

        prediction : Tensor
            Tensor holding the current prediction value.

        Returns
        -------
        Tensor
            Tensor holding the cost between the given input tensors.
        """
        return self._build(target, prediction)

    @abc.abstractmethod
    def _build(self, target, prediction):
        pass


def get_cost(typename):
    """Retrieve Cost class by type

    Parameters
    ----------
    typename : str
        Type of Cost to retrieve

    Returns
    -------
    type
        Cost type found

    Raises
    ------
    ValueError
        When Cost with the given type is not found
    """
    for class_ in luchador.util.get_subclasses(BaseCost):
        if class_.__name__ == typename:
            return class_
    raise ValueError('Unknown Cost: {}'.format(typename))


###############################################################################
# pylint: disable=abstract-method
class BaseSSE(BaseCost):
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
        super(BaseSSE, self).__init__(elementwise=elementwise)


class BaseSigmoidCrossEntropy(BaseCost):
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
        super(BaseSigmoidCrossEntropy, self).__init__(elementwise=elementwise)
