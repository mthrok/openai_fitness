"""Define math ops which work on multiple tensors"""
from __future__ import absolute_import

import theano.tensor as T

from ...wrapper import Tensor

__all__ = ['dot']


def _compute_dot_shape(shape1, shape2):
    if not shape1[-1] == shape2[-2]:
        raise ValueError('Variables not compatible for dot product')
    return shape1[:-1] + shape2[:-2] + shape2[-1:]


def dot(var1, var2, name=None):
    """Implement dot opearation in Theano backend"""
    _tensor = T.dot(var1.unwrap(), var2.unwrap())
    shape = _compute_dot_shape(var1.shape, var2.shape)
    return Tensor(tensor=_tensor, shape=shape, name=name)
