"""Implement clipping methods"""
from __future__ import absolute_import

import tensorflow as tf

from ..wrapper import Tensor

__all__ = ['clip_by_value', 'clip_by_norm']


def clip_by_value(tensor, max_value, min_value, name=None):
    """Implement clip_by_value in Tensorflow backend.

    See :func:`luchador.nn.ops.clip_by_value` for the detail.
    """
    _tensor = tf.clip_by_value(
        tensor.unwrap(), clip_value_min=min_value,
        clip_value_max=max_value, name=name)
    return Tensor(tensor=_tensor, name=name)


def clip_by_norm(tensor, clip_norm, axes=None, name=None):
    """Implement clip_by_norm in Tensorflow backend.

    See :func:`luchador.nn.ops.clip_by_norm` for the detail.
    """
    _tensor = tf.clip_by_norm(
        tensor.unwrap(), clip_norm=clip_norm, axes=axes, name=name)
    return Tensor(tensor=_tensor, name=name)
