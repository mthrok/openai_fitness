"""Define elementwise ops work on multiple tensors"""
from __future__ import absolute_import

from .... import backend as be

__all__ = ['add', 'multiply', 'maximum', 'minimum']


def add(var1, var2, name=None):
    """Elementwise addition with broadcast support

    Parameters
    ----------
    va1, va2 : Tensor
        Tensors to add.

    name : str
        Name of new Tensor

    Returns
    -------
    Tensor
        The resulting Tensor
    """
    return be.ops.add(var1, var2, name=name)


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
    return be.ops.multiply(var1, var2, name=name)


def maximum(var1, var2, name=None):
    """Compute elementwise max among tensors

    Parameters
    ----------
    other : Tensor
        Tensor to compare

    name : str
        Name of new Tensor

    Returns
    -------
    Tensor
        The resulting Tensor
    """
    return be.ops.maximum(var1, var2, name=name)


def minimum(var1, var2, name=None):
    """Compute elementwise min among tensors

    Parameters
    ----------
    var1, var2: Tensor
        Tensor to compare. Either one has to be
        :class:`luchador.nn.theano.wrapper.Tensor` class

    name : str
        Name of new Tensor

    Returns
    -------
    Tensor
        The resulting Tensor
    """
    return be.ops.minimum(var1, var2, name=name)
