from __future__ import absolute_import

import logging

from luchador.common import get_subclasses, StoreMixin

__all__ = [
    'BaseCost', 'get_cost',
    'BaseSSE2',
]

_LG = logging.getLogger(__name__)


class BaseCost(StoreMixin, object):
    """Common interface for cost computation

    Actual Cost class must implement `build` method.
    """
    def __init__(self, elementwise=False, **args):
        """Validate args and set it as instance property. See CopyMixin"""
        super(BaseCost, self).__init__()
        self._store_args(elementwise=elementwise, **args)

    def __call__(self, target, prediction):
        """Build cost between target and prediction

        Args:
          target (Tensor): Correct value.
          prediction (Tensor): The current prediction value.

        Returns:
          Tensor: Resulting cost value
        """
        return self.build(target, prediction)

    def build(self, target, prediction):
        raise NotImplementedError(
            '`build` method is not implemented for {}.{}.'
            .format(type(self).__module__, type(self).__name__)
        )


def get_cost(name):
    for class_ in get_subclasses(BaseCost):
        if class_.__name__ == name:
            return class_
    raise ValueError('Unknown Cost: {}'.format(name))


###############################################################################
class BaseSSE2(BaseCost):
    """Sum-Squared Error

    Actual Cost class must implement `build` method.
    """
    def __init__(self, max_delta=None, min_delta=None, elementwise=False):
        super(BaseSSE2, self).__init__(
            max_delta=max_delta, min_delta=min_delta, elementwise=elementwise)
