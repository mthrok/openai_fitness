"""Implement elementwise ops work on multiple tensors"""
from __future__ import absolute_import

import tensorflow as tf
from ...wrapper import Tensor

__all__ = ['add', 'multiply', 'maximum', 'minimum']


def add(var1, var2, name=None):
    """Implement add"""
    _tensor = tf.add(var1.unwrap(), var2.unwrap(), name=name)
    return Tensor(tensor=_tensor, name=name)


def multiply(var1, var2, name=None):
    """Implement multiply"""
    _tensor = tf.multiply(var1.unwrap(), var2.unwrap(), name=name)
    return Tensor(tensor=_tensor, name=name)


def maximum(var1, var2, name=None):
    """Implement maximum"""
    _tensor = tf.maximum(var1.unwrap(), var2.unwrap(), name=name)
    return Tensor(tensor=_tensor, name=name)


def minimum(var1, var2, name=None):
    """Implement minimum"""
    _tensor = tf.minimum(var1.unwrap(), var2.unwrap(), name=name)
    return Tensor(tensor=_tensor, name=name)
