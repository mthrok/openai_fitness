"""Module for providing backend-common interface for miscellaneous task

Functions in this module are accessible via Anonymous Layer.
"""
from __future__ import absolute_import

from collections import OrderedDict

import numpy as np
import theano.tensor as T

import luchador.util
from luchador.nn.core.base import wrapper as base_wrapper
from .wrapper import Operation, Tensor, Variable

__all__ = [
    'build_sync_op', 'one_hot',
    'abs', 'square', 'sqrt',
    'exp', 'log', 'sin', 'cos',
    'reduce_mean', 'reduce_sum', 'reduce_max',
    'reshape', 'tile', 'add', 'multiply', 'maximum', 'minimum',
    'clip_by_value', 'clip_by_norm',
]
# pylint: disable=redefined-builtin,assignment-from-no-return


def build_sync_op(source_vars, target_vars, tau=None, name='sync'):
    """Build operation to copy values from source variables to target variables

    Parameters
    ----------
    source_vars : list
        list of source variables from which values are copied

    target_vars: list
        list of target variables from which values are copied

    tau : float or None
        Soft assignment parameter. Take weighted sum between target variables
        and source variables;

        .. math::
            target = \\tau source + ( 1 - \\tau)  target

        tau must satisfy 0 <= tau <= 1

        tau == 1 is practically same with tau == None
    """
    if tau is not None:
        if tau < 0 or tau > 1:
            raise ValueError('`tau` must be either None or 0 < tau < 1')

    _operations = OrderedDict()
    for source, target in zip(source_vars, target_vars):
        if not isinstance(target, Variable):
            continue

        src, tgt = source.unwrap(), target.unwrap()
        if tau:
            src = (1 - tau) * tgt + tau * src
        _operations[tgt] = src
    return Operation(op=_operations, name=name)


###############################################################################
def one_hot(var, n_classes, dtype=None, name=None):
    """Convert to one-hot encoding.

    Parameters
    ----------
    n_classes : int
        Number of label to encode

    dtype : str
        The dtype of the resulting Tensor. Default to floatX

    name : str
        Name of operation

    Returns
    -------
    Tensor
        Tensor with shape ``(var.shape[0], n_classes)``

    Notes
    -----
    The Tensor must be either vector or 2D matrix
    """
    if not var.n_dim == 1:
        raise ValueError('Tensor must be 1D.')

    _tensor = T.extra_ops.to_one_hot(var.unwrap(), n_classes, dtype=dtype)
    shape = [var.shape[0], n_classes]
    return Tensor(tensor=_tensor, shape=shape, name=name)


###############################################################################
def _compute_reduced_shape(axis, shape, keep_dims):
    if axis is None:
        if keep_dims:
            return [1] * len(shape)
        return []

    if not luchador.util.is_iteratable(axis):
        axis = [axis]
    if keep_dims:
        return [
            (1 if i in axis else dim)
            for i, dim in enumerate(shape)]
    return [
        dim for i, dim in enumerate(shape)
        if i not in axis]


def _compute_tile_shape(shape, pattern):
    if len(shape) > len(pattern):
        return _compute_tile_shape(pattern, shape)

    _shape = list(pattern)
    offset = len(pattern) - len(shape)
    for i, val in enumerate(shape):
        if _shape[offset + i] is None:
            continue
        if val is not None:
            _shape[offset + i] *= val
    return _shape


def abs(var, name=None):
    """Element-wise absolute value"""
    return var.__abs__(name=name)


def square(var, name=None):
    """Returns square of the given variable"""
    _tensor = T.sqr(var.unwrap())
    return Tensor(tensor=_tensor, shape=var.shape, name=name)


def sqrt(var, name=None):
    """Returns square root of the given variable"""
    _tensor = T.sqrt(var.unwrap())
    return Tensor(tensor=_tensor, shape=var.shape, name=name)


def exp(var, name=None):
    """Returns exponential of the given variable"""
    _tensor = T.exp(var.unwrap())
    return Tensor(tensor=_tensor, shape=var.shape, name=name)


def log(var, name=None):
    """Returns exponential of the given variable"""
    _tensor = T.log(var.unwrap())
    return Tensor(tensor=_tensor, shape=var.shape, name=name)


def sin(var, name=None):
    """Returns sin of the given variable"""
    _tensor = T.sin(var.unwrap())
    return Tensor(tensor=_tensor, shape=var.shape, name=name)


def cos(var, name=None):
    """Returns cos of the given variable"""
    _tensor = T.cos(var.unwrap())
    return Tensor(tensor=_tensor, shape=var.shape, name=name)


def reduce_mean(var, axis=None, keep_dims=False, dtype=None, name=None):
    """Compute mean across the given axis

    Parameters
    ----------
    axis : int, list or None
        The dimensions to compute mean. If None (the default),
        reduces all dimensions.
    keep_dims: bool
        If true, retains reduced dimensions with length 1.
    name: str
        A name for the operation.

    Returns
    -------
    Tensor
        The resulting Tensor
    """
    _tensor = var.unwrap().mean(
        axis=axis, keepdims=keep_dims, dtype=dtype)
    _shape = _compute_reduced_shape(axis, var.shape, keep_dims)
    return Tensor(tensor=_tensor, shape=_shape, name=name)


def reduce_sum(var, axis=None, keep_dims=False, dtype=None, name=None):
    """Compute sum across the given axis

    Parameters
    ----------
    axis : int, list or None
        The dimensions to compute mean. If None (the default),
        reduces all dimensions.
    keep_dims: bool
        If true, retains reduced dimensions with length 1.
    name: str
        A name for the operation.

    Returns
    -------
    Tensor
        The resulting Tensor
    """
    _tensor = var.unwrap().sum(axis=axis, keepdims=keep_dims, dtype=dtype)
    _shape = _compute_reduced_shape(axis, var.shape, keep_dims)
    return Tensor(tensor=_tensor, shape=_shape, name=name)


def reduce_max(var, axis=None, keep_dims=False, name=None):
    """Compute max across the given axis

    Parameters
    ----------
    axis : int, list or None
        The dimensions to compute max. If None (the default),
        reduces all dimensions.
    keep_dims: bool
        If true, retains reduced dimensions with length 1.
    name: str
        A name for the operation.

    Returns
    -------
    Tensor
        The resulting Tensor
    """
    _tensor = var.unwrap().max(axis=axis, keepdims=keep_dims)
    _shape = _compute_reduced_shape(axis, var.shape, keep_dims)
    return Tensor(tensor=_tensor, shape=_shape, name=name)


def _infere_new_shape(original_shape, new_shape):
    if None in original_shape:
        return new_shape

    if -1 in new_shape:
        orig_size = np.prod(original_shape)
        known_size = np.abs(np.prod(new_shape))
        replace = orig_size // known_size
        return tuple(replace if s < 0 else s for s in new_shape)

    return new_shape


def reshape(var, new_shape, name=None):
    """Reshape tensor.

    Parameters
    ----------
    new_shape : tuple
        new shape

    name : str
        Name of operation

    Returns
    -------
    Tensor
        Tensor with new shape

    Notes
    -----
    This function is for conveniently invoke underlying reshap function.
    Shape-checking and inference is not carried out.
    """
    _tensor = T.reshape(var.unwrap(), newshape=new_shape)
    new_shape = _infere_new_shape(var.shape, new_shape)
    return Tensor(tensor=_tensor, shape=new_shape, name=name)


def tile(var, pattern, name=None):
    """Tile tensor.

    Parameters
    ----------
    pattern : tuple
        tile pattern

    name : str
        Name of operation

    Returns
    -------
    Tensor
        Resulting tensor.

    Notes
    -----
    Currently only constant pattern is allowed.
    """
    if not luchador.util.is_iteratable(pattern):
        raise ValueError('`pattern` must be iteratable')

    _shape = _compute_tile_shape(pattern, var.shape)
    _tensor = T.tile(var.unwrap(), pattern)
    return Tensor(tensor=_tensor, shape=_shape, name=name)


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
    pat1, pat2 = _compute_shuffle_pattern(var1.shape, var2.shape)
    var1_ = var1.unwrap().dimshuffle(pat1)
    var2_ = var2.unwrap().dimshuffle(pat2)
    pat1, pat2 = _compute_broadcast_pattern(var1.shape, var2.shape)
    var1_ = T.addbroadcast(var1_, *pat1)
    var2_ = T.addbroadcast(var2_, *pat2)
    shape = _compute_shape(var1.shape, var2.shape)
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
    pat1, pat2 = _compute_shuffle_pattern(var1.shape, var2.shape)
    var1_ = var1.unwrap().dimshuffle(pat1)
    var2_ = var2.unwrap().dimshuffle(pat2)
    pat1, pat2 = _compute_broadcast_pattern(var1.shape, var2.shape)
    var1_ = T.addbroadcast(var1_, *pat1)
    var2_ = T.addbroadcast(var2_, *pat2)
    shape = _compute_shape(var1.shape, var2.shape)
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
    # TODO: Add Broadcasting
    _tensor = T.maximum(var1.unwrap(), var2.unwrap())
    return Tensor(tensor=_tensor, shape=var1.shape, name=name)


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
    # TODO: Add Broadcasting
    _tensor = T.minimum(var1.unwrap(), var2.unwrap())
    return Tensor(tensor=_tensor, shape=var1.shape, name=name)


###############################################################################
def clip_by_value(tensor, max_value, min_value, name=None):
    """Clip value elementwise

    Parameters
    ----------
    max_value, min_value : number or Wrapper
        Clip values

    Returns
    -------
    Tensor
        The resulting Tensor
    """
    if isinstance(max_value, base_wrapper.BaseWrapper):
        max_value = max_value.unwrap()
    if isinstance(min_value, base_wrapper.BaseWrapper):
        min_value = min_value.unwrap()
    _tensor = tensor.unwrap().clip(a_max=max_value, a_min=min_value)
    return Tensor(tensor=_tensor, shape=tensor.shape, name=name)


def clip_by_norm(tensor, clip_norm, axes=None, name=None):
    """Clip tensor values to a maximum L2-norm.

    If the norm of the input ``tensor`` is larger than ``clip_norm``, then
    tensor is rescaled to have norm equals to ``clip_norm``.

    This function is mimic for ``tf.clip_by_norm``. See API documentation
    for the detail.

    Parameters
    ----------
    tensor : Tensor
        Tensor to clip
    clip_norm: A 0-D (scalar) ``Tensor`` > 0. A maximum clipping value.
    axes: A 1-D (vector) ``Tensor`` of type int32 containing the dimensions
      to use for computing the L2-norm. If `None` (the default), uses all
      dimensions.
    name: A name for the operation (optional).

    Returns
    -------
    Tensor
        The resulting Tensor
    """
    shape = tensor.shape
    tensor = tensor.unwrap()
    if isinstance(clip_norm, base_wrapper.BaseWrapper):
        clip_norm = clip_norm.unwrap()
    # Ideally we want do this without unwrapping the variables but
    # `minimum` cannot handle broadcasting yet
    clip_norm_i = 1.0 / clip_norm
    l2norm_i = 1.0 / T.sqrt((tensor * tensor).sum(axis=axes, keepdims=True))
    _tensor = tensor * clip_norm * T.minimum(l2norm_i, clip_norm_i)
    return Tensor(tensor=_tensor, shape=shape, name=name)
