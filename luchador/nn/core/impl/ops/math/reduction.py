"""Define reduction math ops"""
from __future__ import absolute_import

from .... import backend as be

__all__ = ['reduce_mean', 'reduce_sum', 'reduce_max']


def reduce_mean(var, axis=None, keep_dims=False, dtype=None, name=None):
    """Compute mean across the given axis

    Parameters
    ----------
    axis : int, list or None
        The dimensions to compute mean. If None (the default),
        reduces all dimensions.
    keep_dims: bool
        If true, retains reduced dimensions with length 1.
    dtype: str
        Optional in Theano backend. dtype for intermediate computation.
    name: str
        A name for the operation.

    Returns
    -------
    Tensor
        The resulting Tensor
    """
    return be.ops.reduce_mean(
        var, axis=axis, keep_dims=keep_dims, dtype=dtype, name=name)


def reduce_sum(var, axis=None, keep_dims=False, dtype=None, name=None):
    """Compute sum across the given axis

    Parameters
    ----------
    axis : int, list or None
        The dimensions to compute mean. If None (the default),
        reduces all dimensions.
    keep_dims: bool
        If true, retains reduced dimensions with length 1.
    dtype: str
        Optional in Theano backend. dtype for intermediate computation.
    name: str
        A name for the operation.

    Returns
    -------
    Tensor
        The resulting Tensor
    """
    return be.ops.reduce_sum(
        var, axis=axis, keep_dims=keep_dims, dtype=dtype, name=name)


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
    return be.ops.reduce_max(var, axis=axis, keep_dims=keep_dims, name=name)
