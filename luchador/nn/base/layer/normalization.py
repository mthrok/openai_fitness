"""Define common interface for Layer classes"""
from __future__ import division
from __future__ import absolute_import

import logging

from .base import BaseLayer

__all__ = ['BaseBatchNormalization']

_LG = logging.getLogger(__name__)

# pylint: disable=abstract-method


class BaseBatchNormalization(BaseLayer):
    """Apply batch normalization [1]_:

    .. math::
        y = \\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} \\gamma + \\beta

    Notes
    -----
    To fetch paramter variables with :any:`get_variable`, use keys ``mean``,
    ``var``, ``scale`` and ``offset`` in the same scope as layer build.

    To fetch update operation with :any:`get_operation` use key ``bn_update``
    in the same scope as layer build.

    References
    ----------
    .. [1] Ioffe, Sergey and Szegedy, Christian (2015):
           Batch Normalization: Accelerating Deep Network Training by Reducing
           Internal Covariate Shift. http://arxiv.org/abs/1502.03167.

    """
    def __init__(self, scale=1.0, offset=0.0, epsilon=1e-4,
                 learn=True, decay=0.999):
        super(BaseBatchNormalization, self).__init__(
            decay=decay, epsilon=epsilon,
            scale=scale, offset=offset, learn=learn)

        self._create_parameter_slots('mean', 'var', 'scale', 'offset')
        self._axes = self._pattern = None
