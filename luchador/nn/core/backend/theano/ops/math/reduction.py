"""Implement reduction math ops"""
from __future__ import absolute_import

import luchador.util
from ...wrapper import Tensor

__all__ = ['reduce_mean', 'reduce_sum', 'reduce_max']


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
    """Implement reduce_mean"""
    _tensor = var.unwrap().mean(
        axis=axis, keepdims=keep_dims, dtype=dtype)
    _shape = _compute_reduced_shape(axis, var.shape, keep_dims)
    return Tensor(tensor=_tensor, shape=_shape, name=name)


def reduce_sum(var, axis=None, keep_dims=False, dtype=None, name=None):
    """Implement reduce_sum"""
    _tensor = var.unwrap().sum(axis=axis, keepdims=keep_dims, dtype=dtype)
    _shape = _compute_reduced_shape(axis, var.shape, keep_dims)
    return Tensor(tensor=_tensor, shape=_shape, name=name)


def reduce_max(var, axis=None, keep_dims=False, name=None):
    """Implement reduce_max"""
    _tensor = var.unwrap().max(axis=axis, keepdims=keep_dims)
    _shape = _compute_reduced_shape(axis, var.shape, keep_dims)
    return Tensor(tensor=_tensor, shape=_shape, name=name)
