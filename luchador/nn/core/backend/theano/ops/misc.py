"""Define miscellaneous operations"""
from __future__ import absolute_import

from collections import OrderedDict

import theano.tensor as T

from ..wrapper import Operation, Tensor, Variable

__all__ = ['build_sync_op', 'one_hot']


def build_sync_op(source_vars, target_vars, tau=None, name='sync'):
    """Implement ``build_sync_op`` in Theano backend.

    See :func:`luchador.nn.ops.build_sync_op` for the detail.
    """
    _operations = OrderedDict()
    for source, target in zip(source_vars, target_vars):
        if not isinstance(target, Variable):
            continue

        src, tgt = source.unwrap(), target.unwrap()
        if tau:
            src = (1 - tau) * tgt + tau * src
        _operations[tgt] = src
    return Operation(op=_operations, name=name)


def one_hot(var, n_classes, dtype=None, name=None):
    """Implement ``one_hot`` in Theano backend.

    See :func:`luchador.nn.ops.one_hot` for the detail.
    """
    _tensor = T.extra_ops.to_one_hot(var.unwrap(), n_classes, dtype=dtype)
    shape = [var.shape[0], n_classes]
    return Tensor(tensor=_tensor, shape=shape, name=name)
