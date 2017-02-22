
"""Module for providing backend-common interface for misc task"""
from __future__ import absolute_import

from collections import OrderedDict

import theano.tensor as T

from ..base import wrapper as base_wrapper
from .wrapper import Operation, Tensor

__all__ = [
    'build_sync_op', 'one_hot', 'maximum', 'minimum', 'abs',
    'clip_by_value', 'clip_by_norm',
]

# pylint: disable=redefined-builtin


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
def abs(var, name=None):
    """Element-wise absolute value"""
    return var.__abs__(name=name)


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
