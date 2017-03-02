"""Define common interface for Layer classes"""
from __future__ import division
from __future__ import absolute_import

from ...base import BaseLayer
from ...backend import layer

__all__ = ['Add', 'Sub', 'TrueDiv', 'Mean', 'Sin', 'Cos']
# pylint: disable=abstract-method


class Add(layer.Add, BaseLayer):
    """Add tensors

    Parameters
    ----------
    name : str
        Used as base scope when building parameters and output
    """
    def __init__(self, name='Add'):
        super(Add, self).__init__(name=name)


class Sub(layer.Sub, BaseLayer):
    """Subtract tensors

    Parameters
    ----------
    name : str
        Used as base scope when building parameters and output
    """
    def __init__(self, name='Sub'):
        super(Sub, self).__init__(name=name)


class TrueDiv(layer.TrueDiv, BaseLayer):
    """Apply real-valued division to input tensor elementwise

    Parameters
    ----------
    denom : float
        The value of denominator

    name : str
        Used as base scope when building parameters and output
    """
    def __init__(self, denom, name='TrueDiv'):
        super(TrueDiv, self).__init__(denom=denom, name=name)
        self.denom = None


class Mean(layer.Mean, BaseLayer):
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
        super(Mean, self).__init__(
            axis=axis, keep_dims=keep_dims, name=name)


class Sin(layer.Sin, BaseLayer):
    """Apply sin activation elementwise

    Parameters
    ----------
    name : str
        Used as base scope when building parameters and output
    """
    def __init__(self, name='Sin'):
        super(Sin, self).__init__(name=name)


class Cos(layer.Cos, BaseLayer):
    """Apply cos activation elementwise

    Parameters
    ----------
    name : str
        Used as base scope when building parameters and output
    """
    def __init__(self, name='Cos'):
        super(Cos, self).__init__(name=name)
