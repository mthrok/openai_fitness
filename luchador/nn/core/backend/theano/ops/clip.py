"""Implement clipping methods"""
from __future__ import absolute_import

import theano.tensor as T

from ..wrapper import Tensor

__all__ = ['clip_by_value', 'clip_by_norm']


def clip_by_value(tensor, max_value, min_value, name=None):
    """Implement clip_by_value in Theano backend.

    See :func:`luchador.nn.ops.clip_by_value` for the detail.
    """
    _tensor = tensor.unwrap().clip(a_max=max_value, a_min=min_value)
    return Tensor(tensor=_tensor, shape=tensor.shape, name=name)


def clip_by_norm(tensor, clip_norm, axes=None, name=None):
    """Implement clip_by_norm in Theano backend.

    See :func:`luchador.nn.ops.clip_by_norm` for the detail.
    """
    shape = tensor.shape
    tensor = tensor.unwrap()
    # Ideally we want do this without unwrapping the variables but
    # `minimum` cannot handle broadcasting yet
    clip_norm_i = 1.0 / clip_norm
    l2norm_i = 1.0 / T.sqrt((tensor * tensor).sum(axis=axes, keepdims=True))
    _tensor = tensor * clip_norm * T.minimum(l2norm_i, clip_norm_i)
    return Tensor(tensor=_tensor, shape=shape, name=name)
