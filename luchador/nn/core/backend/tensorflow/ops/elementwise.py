"""Define elementwise ops with broadcast support"""
from __future__ import absolute_import

import tensorflow as tf
from ..wrapper import Tensor

__all__ = ['add', 'multiply', 'maximum', 'minimum']


def add(var1, var2, name=None):
    """Elementwise addition with broadcast support

    Parameters
    ----------
    va1, va2 : Tensor
        Tensors to multiply.

    name : str
        Name of new Tensor

    Returns
    -------
    Tensor
        The resulting Tensor
    """
    _tensor = tf.add(var1.unwrap(), var2.unwrap(), name=name)
    return Tensor(tensor=_tensor, name=name)


def multiply(var1, var2, name=None):
    """Elementwise multiplication with broadcast support

    Parameters
    ----------
    va1, va2 : Tensor
        Tensors to multiply.

    name : str
        Name of new Tensor

    Returns
    -------
    Tensor
        The resulting Tensor
    """
    _tensor = tf.multiply(var1.unwrap(), var2.unwrap(), name=name)
    return Tensor(tensor=_tensor, name=name)


def maximum(var1, var2, name=None):
    """Compute elementwise max against other tensor

    Parameters
    ----------
    var1, var2 : Tensor
        Tensors to compare. In Tensorflow backend, the shape of other
        Tensor can be different as long as it is broadcastable.

    name : str
        Name of new Tensor

    Returns
    -------
    Tensor
        The resulting Tensor
    """
    _tensor = tf.maximum(var1.unwrap(), var2.unwrap(), name=name)
    return Tensor(tensor=_tensor, name=name)


def minimum(var1, var2, name=None):
    """Compute elementwise min against other tensor

    Parameters
    ----------
    va1, var2 : Tensor
        Tensors to compare. In Tensorflow backend, the shape of other
        Tensor can be different as long as it is broadcastable.

    name : str
        Name of new Tensor

    Returns
    -------
    Tensor
        The resulting Tensor
    """
    _tensor = tf.minimum(var1.unwrap(), var2.unwrap(), name=name)
    return Tensor(tensor=_tensor, name=name)
