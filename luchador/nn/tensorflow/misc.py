"""Module for providing backend-common interface for misc task"""
from __future__ import absolute_import

import tensorflow as tf

from .wrapper import Operation, Tensor

__all__ = ['build_sync_op', 'mean']


def build_sync_op(source_vars, target_vars, name='sync'):
    """Build operation to copy values"""
    op = [
        target.unwrap().assign(source.unwrap())
        for source, target in zip(source_vars, target_vars)
    ]
    return Operation(op=op, name=name)


def mean(tensor, axis, keep_dims=False, name=None):
    """Compute mean"""
    # pylint: disable=protected-access
    _tensor = tf.reduce_mean(
        tensor._tensor, axis=axis, keep_dims=keep_dims, name=name)
    return Tensor(tensor=_tensor, name=name)
