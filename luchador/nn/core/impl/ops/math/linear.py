"""Define math ops which work on multiple tensors"""
from __future__ import absolute_import

from .... import backend as be

__all__ = ['dot']


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
    return be.ops.dot(var1, var2, name=name)
