"""Module for providing backend-common interface for misc task"""
from __future__ import absolute_import

from collections import OrderedDict

import luchador.util
from .wrapper import Operation, Tensor

__all__ = ['build_sync_op', 'mean']


def build_sync_op(source_vars, target_vars, name='sync'):
    """Build operation to copy values"""
    op = OrderedDict()
    for src, tgt in zip(source_vars, target_vars):
        op[tgt.unwrap()] = src.unwrap()
    return Operation(op=op, name=name)


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


def mean(tensor, axis, keep_dims=False, dtype=None, name=None):
    """Compute mean"""
    # pylint: disable=protected-access
    _tensor = tensor._tensor.mean(axis=axis, keepdims=keep_dims, dtype=dtype)
    _shape = _compute_reduced_shape(axis, tensor.shape, keep_dims)
    return Tensor(tensor=_tensor, shape=_shape, name=name)
