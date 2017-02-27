"""Define common interface for Layer classes"""
from __future__ import division
from __future__ import absolute_import

import logging

from .base import BaseLayer

__all__ = [
    'BaseAdd', 'BaseSub', 'BaseTrueDiv', 'BaseMean', 'BaseSin', 'BaseCos'
]

_LG = logging.getLogger(__name__)

# pylint: disable=abstract-method


class BaseAdd(BaseLayer):
    """Add tensors

    Parameters
    ----------
    name : str
        Used as base scope when building parameters and output
    """
    def __init__(self, name='Add'):
        super(BaseAdd, self).__init__(name=name)


class BaseSub(BaseLayer):
    """Subtract tensors

    Parameters
    ----------
    name : str
        Used as base scope when building parameters and output
    """
    def __init__(self, name='Sub'):
        super(BaseSub, self).__init__(name=name)


class BaseTrueDiv(BaseLayer):
    """Apply real-valued division to input tensor elementwise

    Parameters
    ----------
    denom : float
        The value of denominator

    name : str
        Used as base scope when building parameters and output
    """
    def __init__(self, denom, name='TrueDiv'):
        super(BaseTrueDiv, self).__init__(denom=denom, name=name)
        self.denom = None


class BaseMean(BaseLayer):
    """Apply mean to input tensor

    Parameters
    ----------
    axis : int or list of int
        Axis or axes along which to compute the mean
    keep_dim : bool
        If true, retains reduced dimensions with length 1.
    dtype : str
        Output dtype
    name : str
        Used as base scope when building parameters and output
    """
    def __init__(self, axis, keep_dims=False, name='Mean'):
        super(BaseMean, self).__init__(
            axis=axis, keep_dims=keep_dims, name=name)


class BaseSin(BaseLayer):
    """Apply sin activation elementwise

    Parameters
    ----------
    name : str
        Used as base scope when building parameters and output
    """
    def __init__(self, name='Sin'):
        super(BaseSin, self).__init__(name=name)


class BaseCos(BaseLayer):
    """Apply cos activation elementwise

    Parameters
    ----------
    name : str
        Used as base scope when building parameters and output
    """
    def __init__(self, name='Cos'):
        super(BaseCos, self).__init__(name=name)
