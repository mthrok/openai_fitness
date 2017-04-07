"""Define transform operations"""
from __future__ import absolute_import

import tensorflow as tf

from ..wrapper import Tensor

__all__ = ['reshape', 'tile']


def reshape(var, new_shape, name=None):
    """Implement ``reshape`` in Tensorflow backend.

    See :func:`luchador.nn.ops.reshape` for detail
    """
    _tensor = tf.reshape(var.unwrap(), shape=new_shape)
    return Tensor(tensor=_tensor, name=name)


def tile(var, pattern, name=None):
    """Implement ``tile`` in Tensorflow backend.

    See :func:`luchador.nn.ops.tile` for detail
    """
    if len(pattern) > var.n_dim:
        prepend = (1, ) * (len(pattern) - var.n_dim)
        tensor = reshape(var, prepend + var.shape).unwrap()
    else:
        prepend = (1, ) * (var.n_dim - len(pattern))
        pattern = prepend + pattern
        tensor = var.unwrap()
    return Tensor(tf.tile(tensor, pattern, name), name=name)
