"""Define common interface for Layer classes"""
from __future__ import division
from __future__ import absolute_import

from ...base import BaseLayer
from ...backend import layer

__all__ = ['TrueDiv']
# pylint: disable=abstract-method


class TrueDiv(layer.TrueDiv, BaseLayer):
    """Apply real-valued division to input tensor elementwise

    Parameters
    ----------
    denom : float
        The value of denominator

    scope : str
        Used as base scope when building parameters and output
    """
    def __init__(self, denom, scope='TrueDiv'):
        super(TrueDiv, self).__init__(denom=denom, scope=scope)
        self._denom = None
