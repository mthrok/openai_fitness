"""Implement reduction math ops"""
from __future__ import absolute_import

import tensorflow as tf

from ...wrapper import Tensor

__all__ = ['reduce_mean', 'reduce_sum', 'reduce_max']
# pylint: disable=unused-argument


def reduce_mean(var, axis=None, keep_dims=False, dtype=None, name=None):
    """Implement reduce_mean"""
    _tensor = tf.reduce_mean(
        var.unwrap(), axis=axis, keep_dims=keep_dims, name=name)
    return Tensor(tensor=_tensor, name=name)


def reduce_sum(var, axis=None, keep_dims=False, dtype=None, name=None):
    """Implement reduce_sum"""
    _tensor = tf.reduce_sum(
        var.unwrap(), axis=axis, keep_dims=keep_dims, name=name)
    return Tensor(tensor=_tensor, name=name)


def reduce_max(var, axis=None, keep_dims=False, name=None):
    """Implement reduce_max"""
    _tensor = tf.reduce_max(
        var.unwrap(), axis=axis, keep_dims=keep_dims, name=name)
    return Tensor(tensor=_tensor, name=name)
