"""Define gradient-related operations"""
from __future__ import absolute_import

import logging

import theano

from ..wrapper import Tensor

__all__ = ['compute_gradient']
_LG = logging.getLogger(__name__)


def compute_gradient(loss, wrt, **kwargs):
    """Implement ``compute_gradient`` in Theano backend.

    See :func:`luchador.nn.ops.compute_gradient` for detail
    """
    # So as to match the behavior to that of Tensorflow, we return None
    # for disconnected inputs
    grads = theano.grad(
        loss.unwrap(), wrt, disconnected_inputs='warn',
        return_disconnected='None', **kwargs)
    ret, i = [], 0
    for var in wrt:
        tensor = None
        if var.trainable:
            grad = grads[i]
            i += 1
            if grad is not None:
                name_ = '{}_grad'.format(var.name)
                tensor = Tensor(grad, shape=var.shape, name=name_)
        ret.append((tensor, var))
    return ret
