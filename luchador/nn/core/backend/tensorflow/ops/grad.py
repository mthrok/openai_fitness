"""Define gradient-related operations"""
from __future__ import absolute_import

import logging

import tensorflow as tf

from luchador.util import is_iteratable
from ..wrapper import Tensor

__all__ = ['compute_gradient']
_LG = logging.getLogger(__name__)


def compute_gradient(loss, wrt, **kwargs):
    """Implement ``compute_gradient`` in Tensorflow backend.

    See :func:`luchador.nn.ops.compute_gradient` for detail
    """
    _LG.info('Computing gradient for %s', loss)
    wrt = wrt if is_iteratable(wrt) else [wrt]
    for var in wrt:
        _LG.info('    %20s', var)
    wrt_ = [v.unwrap() for v in wrt if v.trainable]

    if not wrt_:
        raise ValueError('No variables to optimize.')

    ys = loss.unwrap()
    grads = tf.gradients(ys=ys, xs=wrt_, **kwargs)
    ret, i = [], 0
    for var in wrt:
        tensor = None
        if var.trainable:
            grad = grads[i]
            i += 1
            if grad is not None:
                name_ = '{}_grad'.format(var.name)
                tensor = Tensor(grad, name=name_)
        ret.append((tensor, var))
    return ret
