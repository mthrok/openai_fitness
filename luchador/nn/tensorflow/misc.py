"""Module for providing backend-common interface for misc task"""
from __future__ import absolute_import

import tensorflow as tf

import luchador
from .wrapper import Operation, Tensor

__all__ = ['build_sync_op', 'one_hot', 'maximum', 'minimum']


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

    _operations = []
    for source, target in zip(source_vars, target_vars):
        src, tgt = source.unwrap(), target.unwrap()
        if tau:
            src = (1 - tau) * tgt + tau * src
        _operations.append(tgt.assign(src))
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

    _dtype = dtype or luchador.get_nn_dtype()
    _tensor = tf.one_hot(
        var.unwrap(), depth=n_classes, dtype=_dtype, name=name)
    return Tensor(tensor=_tensor, name=name)


###############################################################################
def maximum(var1, var2, name=None):
    """Compute elementwise max against other tensor

    Parameters
    ----------
    other : Tensor
        Tensor to compare. In Tensorflow backend, the shape of other
        Tensor can be different as long as it is broadcastable.

    name : str
        Name of new Tensor

    Returns
    -------
    Tensor
        The resulting Tensor
    """
    _tensor = tf.maximum(var1.unwrap(), var2.unwrap(), name=name)
    return Tensor(tensor=_tensor, name=name)


def minimum(var1, var2, name=None):
    """Compute elementwise min against other tensor

    Parameters
    ----------
    other : Tensor
        Tensor to compare. In Tensorflow backend, the shape of other
        Tensor can be different as long as it is broadcastable.

    name : str
        Name of new Tensor

    Returns
    -------
    Tensor
        The resulting Tensor
    """
    _tensor = tf.minimum(var1.unwrap(), var2.unwrap(), name=name)
    return Tensor(tensor=_tensor, name=name)
