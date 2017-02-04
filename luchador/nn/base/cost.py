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
class BaseSSE2(BaseCost):
    """Compute Sum-Squared-Error / 2.0 for the given target and prediction

    TODO: Add math expression

    Parameters
    ----------
    max_delta, min_delta : float
        Clip the difference between target value and prediction values

    elementwise : Bool
        When true, the cost tesnor returned by this method has the same shape
        as its input Tensors. When False, the cost tensor is flattened
        to scalar shape by taking average over batch and sum over feature.
        Default: False.

    Notes
    -----
    In case ``elementwise=True``, this cost is reduced to squared difference
    between target and prediction.
    """
    def __init__(self, max_delta=None, min_delta=None, elementwise=False):
        super(BaseSSE2, self).__init__(
            max_delta=max_delta, min_delta=min_delta, elementwise=elementwise)

    def _validate_args(self, min_delta, max_delta, **_):
        """Check if constructor arguments are valid. Raise error if invalid.

        Called automatically by constructor. Not to be called by user.
        """
        if (min_delta and max_delta) or (not max_delta and not min_delta):
            return
        raise ValueError('When clipping delta, both '
                         '`min_delta` and `max_delta` must be provided')


class BaseSigmoidCrossEntropy(BaseCost):
    """Directory computes classification entropy from logit

    Parameters
    ----------
    elementwise : Bool
        When True, the cost tesnor returned by `build` method has the same
        shape as its input Tensors. When False, the cost tensor is flattened
        to scalar shape by taking average over batch and sum over feature.
        Defalut: False.
    """
    def __init__(self, elementwise=False):
        super(BaseSigmoidCrossEntropy, self).__init__(elementwise=elementwise)
