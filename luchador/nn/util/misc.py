"""Misc operations which can be implemented with common backend interface"""
from __future__ import absolute_import

import luchador.nn

__all__ = ['clip_grads']


def clip_grads(grads_and_vars, max_value, min_value):
    """Clip gradients

    Parameters
    ----------
    grads_and_vars : list
        Gradient and Variable tuples. Return value from ``compute_gradients``.

    max_value, min_value : Number
        Value to clip gradients

    Returns
    -------
    list
        Resulting gradients and vars pairs
    """
    ret = []
    for grad, var in grads_and_vars:
        name = '{}_clip'.format(grad.name)
        grad = luchador.nn.backend.clip_by_value(
            grad, max_value=max_value, min_value=min_value, name=name)
        ret.append((grad, var))
    return ret
