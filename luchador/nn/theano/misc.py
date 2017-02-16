"""Module for providing backend-common interface for misc task"""
from __future__ import absolute_import

from collections import OrderedDict

import theano.tensor as T

import luchador.util
from ..base.wrapper import BaseWrapper
from .wrapper import Operation, Tensor

__all__ = ['build_sync_op']

# pylint: disable=redefined-builtin


def build_sync_op(source_vars, target_vars, tau=None, name='sync'):
    """Build operation to copy values from source variables to target variables

    Parameters
    ----------
    source_vars : list
        list of source variables from which values are copied

    target_vars: list
        list of target variables from which values are copied

    tau : float or None
        Soft assignment parameter. Take weighted sum between target variables
        and source variables;

        .. math::
            target = \\tau source + ( 1 - \\tau)  target

        tau must satisfy 0 <= tau <= 1

        tau == 1 is practically same with tau == None
    """
    if tau is not None:
        if tau < 0 or tau > 1:
            raise ValueError('`tau` must be either None or 0 < tau < 1')

    _operations = OrderedDict()
    for source, target in zip(source_vars, target_vars):
        src, tgt = source.unwrap(), target.unwrap()
        if tau:
            src = (1 - tau) * tgt + tau * src
        _operations[tgt] = src
    return Operation(op=_operations, name=name)


###############################################################################
def _compute_reduced_shape(axis, shape, keep_dims):
    if not luchador.util.is_iteratable(axis):
        axis = [axis]
    if keep_dims:
        return [
            (1 if i in axis else dim)
            for i, dim in enumerate(shape)]
    return [
        dim for i, dim in enumerate(shape)
        if i not in axis]


def mean(var, axis=None, keep_dims=False, dtype=None, name=None):
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
    _tensor = var.unwrap().mean(axis=axis, keepdims=keep_dims, dtype=dtype)
    _shape = _compute_reduced_shape(axis, var.shape, keep_dims)
    return Tensor(tensor=_tensor, shape=_shape, name=name)


def sum(var, axis=None, keep_dims=False, dtype=None, name=None):
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


def max(var, axis=None, keep_dims=False, name=None):
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


def clip(var, max_value, min_value, name=None):
    """Clip value elementwise

    Parameters
    ----------
    max_value, min_value : number or Wrapper
        Clip values

    Returns
    -------
    Tensor
        The resulting Tensor
    """
    if isinstance(max_value, BaseWrapper):
        max_value = max_value.unwrap()
    if isinstance(min_value, BaseWrapper):
        min_value = min_value.unwrap()
    _tensor = var.unwrap().clip(a_max=max_value, a_min=min_value)
    return Tensor(tensor=_tensor, shape=var.shape, name=name)


###############################################################################
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

    Notes
    -----
    This function is for conveniently invoke underlying reshap function.
    Shape-checking and inference is not carried out.
    """
    _tensor = T.reshape(var.unwrap(), newshape=new_shape)
    return Tensor(tensor=_tensor, shape=new_shape, name=name)


def _compute_tile_shape(shape, pattern):
    if len(shape) > len(pattern):
        return _compute_tile_shape(pattern, shape)

    _shape = list(pattern)
    offset = len(pattern) - len(shape)
    for i, val in enumerate(shape):
        if _shape[offset + i] is None:
            continue
        if val is not None:
            _shape[offset + i] *= val
    return _shape


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
    if not luchador.util.is_iteratable(pattern):
        raise ValueError('`pattern` must be iteratable')

    _shape = _compute_tile_shape(pattern, var.shape)
    _tensor = T.tile(var.unwrap(), pattern)
    return Tensor(tensor=_tensor, shape=_shape, name=name)


###############################################################################
def one_hot(var, n_classes, dtype=None, name=None):
    """Convert to one-hot encoding.

    Parameters
    ----------
    n_classes : int
        Number of label to encode

    dtype : str
        The dtype of the resulting Tensor. Default to floatX

    name : str
        Name of operation

    Returns
    -------
    Tensor
        Tensor with shape ``(var.shape[0], n_classes)``

    Notes
    -----
    The Tensor must be either vector or 2D matrix
    """
    if not var.n_dim == 1:
        raise ValueError('Tensor must be 1D.')

    _tensor = T.extra_ops.to_one_hot(var.unwrap(), n_classes, dtype=dtype)
    shape = [var.shape[0], n_classes]
    return Tensor(tensor=_tensor, shape=shape, name=name)


###############################################################################
def maximum(var1, var2, name=None):
    """Compute elementwise max among tensors

    Parameters
    ----------
    other : Tensor
        Tensor to compare

    name : str
        Name of new Tensor

    Returns
    -------
    Tensor
        The resulting Tensor
    """
    # TODO: Add Broadcasting
    _tensor = T.maximum(var1.unwrap(), var2.unwrap())
    return Tensor(tensor=_tensor, shape=var1.shape, name=name)


def minimum(var1, var2, name=None):
    """Compute elementwise min among tensors

    Parameters
    ----------
    other : Tensor
        Tensor to compare

    name : str
        Name of new Tensor

    Returns
    -------
    Tensor
        The resulting Tensor
    """
    # TODO: Add Broadcasting
    _tensor = T.minimum(var1.unwrap(), var2.unwrap())
    return Tensor(tensor=_tensor, shape=var1.shape, name=name)
