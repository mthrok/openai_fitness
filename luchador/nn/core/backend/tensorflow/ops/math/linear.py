"""Define math ops which work on multiple tensors"""
from __future__ import absolute_import

import tensorflow as tf

from ...wrapper import Tensor

__all__ = ['dot']


def dot(var1, var2, name=None):
    """Compute dot product of input variables

    Parameters
    ----------
    var1, var2 : Variable
        Variables to be multipled. The last dimension of var1 must match
        the 2nd to the last dimension of var2.

    name : str
        The name of the resulting tensor

    Return
    ------
    Tensor
        The resulting Tensor
    """
    axes = [[var1.n_dim-1], [var2.n_dim-2]]
    _tensor = tf.tensordot(var1.unwrap(), var2.unwrap(), axes=axes, name=name)
    return Tensor(tensor=_tensor, name=name)
