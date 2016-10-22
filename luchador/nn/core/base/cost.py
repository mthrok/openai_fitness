from __future__ import absolute_import

import logging

from luchador.common import StoreMixin

__all__ = [
    'BaseCost', 'BaseSSE2'
]

_LG = logging.getLogger(__name__)


class BaseCost(StoreMixin, object):
    """Common interface for cost computation

    Actual Cost class must implement `build` method.
    """
    def __init__(self, **args):
        """Validate args and set it as instance property. See CopyMixin"""
        super(BaseCost, self).__init__()
        self._store_args(**args)

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


class BaseSSE2(BaseCost):
    """Sum-Squared Error

    Actual Cost class must implement `build` method.
    """
    def __init__(self, max_delta=None, min_delta=None):
        super(BaseSSE2, self).__init__(
            max_delta=max_delta, min_delta=min_delta)
