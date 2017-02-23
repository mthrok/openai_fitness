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
    """Add tensors"""
    def __init__(self):
        super(BaseAdd, self).__init__()


class BaseSub(BaseLayer):
    """Subtract tensors"""
    def __init__(self):
        super(BaseSub, self).__init__()


class BaseTrueDiv(BaseLayer):
    """Apply real-valued division to input tensor elementwise

    Parameters
    ----------
    denom : float
        The value of denominator
    """
    def __init__(self, denom):
        super(BaseTrueDiv, self).__init__(denom=denom)
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
    """
    def __init__(self, axis, keep_dims=False):
        super(BaseMean, self).__init__(axis=axis, keep_dims=keep_dims)


class BaseSin(BaseLayer):
    """Apply sin activation elementwise"""
    def __init__(self):
        super(BaseSin, self).__init__()


class BaseCos(BaseLayer):
    """Apply cos activation elementwise"""
    def __init__(self):
        super(BaseCos, self).__init__()
