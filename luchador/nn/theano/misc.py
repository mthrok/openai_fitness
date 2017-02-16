"""Module for providing backend-common interface for misc task"""
from __future__ import absolute_import

from collections import OrderedDict

import theano.tensor as T

from .wrapper import Operation, Tensor

__all__ = ['build_sync_op', 'one_hot', 'maximum', 'minimum']

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
def abs(var, name):
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
    _tensor = T.minimum(var1.unwrap(), var2.unwrap())
    return Tensor(tensor=_tensor, shape=var1.shape, name=name)
