"""Define miscellaneous operations"""
from __future__ import absolute_import

import tensorflow as tf

import luchador
from ..wrapper import Variable, Tensor, Operation

__all__ = ['build_sync_op', 'one_hot']


def build_sync_op(source_vars, target_vars, tau=None, name='sync'):
    """Implement ``build_sync_op`` in Tensorflow backend.

    See :func:`luchador.nn.ops.build_sync_op` for the detail.
    """
    _operations = []
    for source, target in zip(source_vars, target_vars):
        if not isinstance(target, Variable):
            continue

        src, tgt = source.unwrap(), target.unwrap()
        if tau:
            src = (1 - tau) * tgt + tau * src
        _operations.append(tgt.assign(src))
    return Operation(op=_operations, name=name)


def one_hot(var, n_classes, dtype=None, name=None):
    """Implement ``one_hot`` in Tensorflow backend.

    See :func:`luchador.nn.ops.one_hot` for the detail.
    """
    _dtype = dtype or luchador.get_nn_dtype()
    _tensor = tf.one_hot(
        var.unwrap(), depth=n_classes, dtype=_dtype, name=name)
    return Tensor(tensor=_tensor, name=name)
