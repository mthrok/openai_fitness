"""Define transform operations"""
from __future__ import absolute_import

from luchador.util import is_iteratable
from ... import backend as be

__all__ = ['reshape', 'tile']


def reshape(var, new_shape, name=None):
    """Reshape tensor.

    Parameters
    ----------
    new_shape : tuple
        new shape

    name : str
        Name of operation

    Returns
    -------
    Tensor
        Tensor with new shape
    """
    return be.ops.reshape(var, new_shape, name)


def tile(var, pattern, name=None):
    """Tile tensor.

    Parameters
    ----------
    pattern : tuple
        tile pattern

    name : str
        Name of operation

    Returns
    -------
    Tensor
        Resulting tensor.

    Notes
    -----
    Currently only constant pattern is allowed.
    """
    if not is_iteratable(pattern):
        raise ValueError('`pattern` must be iteratable')
    return be.ops.tile(var, tuple(pattern), name)
