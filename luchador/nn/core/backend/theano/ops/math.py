"""Define math ops"""
from __future__ import absolute_import

import theano.tensor as T

import luchador.util
from ..wrapper import Tensor

__all__ = [
    'abs', 'square', 'sqrt',
    'exp', 'log', 'sin', 'cos',
    'reduce_mean', 'reduce_sum', 'reduce_max',
]
# pylint: disable=redefined-builtin,assignment-from-no-return


def abs(var, name=None):
    """Element-wise absolute value"""
    return var.__abs__(name=name)


def square(var, name=None):
    """Returns square of the given variable"""
    _tensor = T.sqr(var.unwrap())
    return Tensor(tensor=_tensor, shape=var.shape, name=name)


def sqrt(var, name=None):
    """Returns square root of the given variable"""
    _tensor = T.sqrt(var.unwrap())
    return Tensor(tensor=_tensor, shape=var.shape, name=name)


def exp(var, name=None):
    """Returns exponential of the given variable"""
    _tensor = T.exp(var.unwrap())
    return Tensor(tensor=_tensor, shape=var.shape, name=name)


def log(var, name=None):
    """Returns exponential of the given variable"""
    _tensor = T.log(var.unwrap())
    return Tensor(tensor=_tensor, shape=var.shape, name=name)


def sin(var, name=None):
    """Returns sin of the given variable"""
    _tensor = T.sin(var.unwrap())
    return Tensor(tensor=_tensor, shape=var.shape, name=name)


def cos(var, name=None):
    """Returns cos of the given variable"""
    _tensor = T.cos(var.unwrap())
    return Tensor(tensor=_tensor, shape=var.shape, name=name)


###############################################################################
def _compute_reduced_shape(axis, shape, keep_dims):
    if axis is None:
        if keep_dims:
            return [1] * len(shape)
        return []

    if not luchador.util.is_iteratable(axis):
        axis = [axis]
    if keep_dims:
        return [
            (1 if i in axis else dim)
            for i, dim in enumerate(shape)]
    return [
        dim for i, dim in enumerate(shape)
        if i not in axis]


def reduce_mean(var, axis=None, keep_dims=False, dtype=None, name=None):
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
    _tensor = var.unwrap().mean(
        axis=axis, keepdims=keep_dims, dtype=dtype)
    _shape = _compute_reduced_shape(axis, var.shape, keep_dims)
    return Tensor(tensor=_tensor, shape=_shape, name=name)


def reduce_sum(var, axis=None, keep_dims=False, dtype=None, name=None):
    """Compute sum across the given axis

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
    _tensor = var.unwrap().sum(axis=axis, keepdims=keep_dims, dtype=dtype)
    _shape = _compute_reduced_shape(axis, var.shape, keep_dims)
    return Tensor(tensor=_tensor, shape=_shape, name=name)


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
    _tensor = var.unwrap().max(axis=axis, keepdims=keep_dims)
    _shape = _compute_reduced_shape(axis, var.shape, keep_dims)
    return Tensor(tensor=_tensor, shape=_shape, name=name)
