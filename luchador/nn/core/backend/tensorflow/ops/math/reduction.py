"""Define reduction math ops"""
from __future__ import absolute_import

import tensorflow as tf

from ...wrapper import Tensor

__all__ = ['reduce_mean', 'reduce_sum', 'reduce_max']


def reduce_mean(var, axis=None, keep_dims=False, name=None):
    """Compute mean across the given axis

    Parameters
    ----------
    axis : int, list or None
        The dimensions to compute mean. If None (the default),
        reduces all dimensions.
    keep_dims: bool
        If true, retains reduced dimensions with length 1.
    name: str
        A name for the operation.

    Returns
    -------
    Tensor
        The resulting Tensor
    """
    _tensor = tf.reduce_mean(
        var.unwrap(), axis=axis, keep_dims=keep_dims, name=name)
    return Tensor(tensor=_tensor, name=name)


def reduce_sum(var, axis=None, keep_dims=False, name=None):
    """Compute sum across the given axis

    Parameters
    ----------
    axis : int, list or None
        The dimensions to compute sum. If None (the default),
        reduces all dimensions.
    keep_dims: bool
        If true, retains reduced dimensions with length 1.
    name: str
        A name for the operation.

    Returns
    -------
    Tensor
        The resulting Tensor
    """
    _tensor = tf.reduce_sum(
        var.unwrap(), axis=axis, keep_dims=keep_dims, name=name)
    return Tensor(tensor=_tensor, name=name)


def reduce_max(var, axis=None, keep_dims=False, name=None):
    """Compute max across the given axis

    Parameters
    ----------
    axis : int, list or None
        The dimensions to compute max. If None (the default),
        reduces all dimensions.
    keep_dims: bool
        If true, retains reduced dimensions with length 1.
    name: str
        A name for the operation.

    Returns
    -------
    Tensor
        The resulting Tensor
    """
    _tensor = tf.reduce_max(
        var.unwrap(), axis=axis, keep_dims=keep_dims, name=name)
    return Tensor(tensor=_tensor, name=name)
