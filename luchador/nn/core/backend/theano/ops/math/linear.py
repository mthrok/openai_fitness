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
    """Compute dot product of input variables ``var1 * var2``

    Parameters
    ----------
    var1, var2 : Variable
        Variables to be multipled. The last dimension of var1 must match
        the 2nd to thes last dimension of var2.

    name : str
        The name of the resulting tensor

    Return
    ------
    Tensor
        The resulting Tensor
    """
    _tensor = T.dot(var1.unwrap(), var2.unwrap())
    shape = _compute_dot_shape(var1.shape, var2.shape)
    return Tensor(tensor=_tensor, shape=shape, name=name)
