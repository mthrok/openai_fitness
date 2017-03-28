"""Define elementwise ops work on multiple tensors"""
from __future__ import absolute_import

import theano.tensor as T

from ...wrapper import Tensor

__all__ = ['add', 'multiply', 'maximum', 'minimum']
# pylint: disable=assignment-from-no-return


def _compute_shuffle_pattern(shape1, shape2):
    dim1, dim2 = len(shape1), len(shape2)
    if dim1 < dim2:
        ret = _compute_shuffle_pattern(shape2, shape1)
        return ret[1], ret[0]

    diff = dim1 - dim2
    pat1 = [i for i in range(diff)]
    pat2 = ['x' for _ in range(diff)]
    for i1 in range(diff, dim1):
        i2 = i1 - diff
        pat1.append(i1)
        pat2.append(i2)
    return pat1, pat2


def _compute_broadcast_pattern(shape1, shape2):
    dim1, dim2 = len(shape1), len(shape2)
    if dim1 < dim2:
        ret = _compute_broadcast_pattern(shape2, shape1)
        return ret[1], ret[0]

    diff = dim1 - dim2
    pat1 = []
    pat2 = [i for i in range(diff)]
    for i1 in range(diff, dim1):
        i2 = i1 - diff
        if shape1[i1] == shape2[i2]:
            pass
        elif shape1[i1] == 1:
            pat1.append(i1)
        elif shape2[i2] == 1:
            pat2.append(i1)
    return pat1, pat2


def _compute_shape(shape1, shape2):
    dim1, dim2 = len(shape1), len(shape2)
    if dim1 < dim2:
        return _compute_shape(shape2, shape1)

    diff = dim1 - dim2
    shape = list(shape1[:diff])
    for i1 in range(diff, dim1):
        i2 = i1 - diff
        if shape1[i1] == shape2[i2]:
            shape.append(shape1[i1])
        elif shape1[i1] == 1 or shape2[i2] is None:
            shape.append(shape2[i2])
        elif shape2[i2] == 1 or shape1[i1] is None:
            shape.append(shape1[i1])
        else:
            raise ValueError('Incompatible shape')
    return tuple(shape)


def _make_compatible(var1, var2):
    """Align dimensions and broadcast pattern of the input variables"""
    pat1, pat2 = _compute_shuffle_pattern(var1.shape, var2.shape)
    var1_ = var1.unwrap().dimshuffle(pat1)
    var2_ = var2.unwrap().dimshuffle(pat2)
    pat1, pat2 = _compute_broadcast_pattern(var1.shape, var2.shape)
    var1_ = T.addbroadcast(var1_, *pat1)
    var2_ = T.addbroadcast(var2_, *pat2)
    shape = _compute_shape(var1.shape, var2.shape)
    return var1_, var2_, shape


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
    var1_, var2_, shape = _make_compatible(var1, var2)
    return Tensor(tensor=var1_+var2_, shape=shape, name=name)


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
    var1_, var2_, shape = _make_compatible(var1, var2)
    return Tensor(tensor=var1_*var2_, shape=shape, name=name)


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
    var1_, var2_, shape = _make_compatible(var1, var2)
    _tensor = T.maximum(var1_, var2_)
    return Tensor(tensor=_tensor, shape=shape, name=name)


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
    var1_, var2_, shape = _make_compatible(var1, var2)
    _tensor = T.minimum(var1_, var2_)
    return Tensor(tensor=_tensor, shape=shape, name=name)
