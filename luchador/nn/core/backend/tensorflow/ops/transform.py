"""Define transform operations"""
from __future__ import absolute_import

import tensorflow as tf

from ..wrapper import Tensor

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
    _tensor = tf.reshape(var.unwrap(), shape=new_shape)
    return Tensor(tensor=_tensor, name=name)


def tile(var, pattern, name=None):
    """Tile tensor.

    Parameters
    ----------
    pattern : tuple
        tile pattern

    Notes
    -----
    Currently only constant pattern is allowed.
    """
    try:
        pattern = tuple(pattern)
    except TypeError:
        raise ValueError('`pattern` must be iteratable')

    if len(pattern) > var.n_dim:
        prepend = (1, ) * (len(pattern) - var.n_dim)
        tensor = reshape(var, prepend + var.shape).unwrap()
    else:
        prepend = (1, ) * (var.n_dim - len(pattern))
        pattern = prepend + pattern
        tensor = var.unwrap()
    return Tensor(tf.tile(tensor, pattern, name), name=name)
