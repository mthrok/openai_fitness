"""Define clipping methods"""
from __future__ import absolute_import

import tensorflow as tf

from luchador.nn.core.base import BaseWrapper
from ..wrapper import Tensor

__all__ = ['clip_by_value', 'clip_by_norm']


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
    if isinstance(max_value, BaseWrapper):
        max_value = max_value.unwrap()
    if isinstance(min_value, BaseWrapper):
        min_value = min_value.unwrap()
    _tensor = tf.clip_by_value(
        tensor.unwrap(), clip_value_min=min_value,
        clip_value_max=max_value, name=name)
    return Tensor(tensor=_tensor, name=name)


def clip_by_norm(tensor, clip_norm, axes=None, name=None):
    """Clip tensor values to a maximum L2-norm.

    If the norm of the input ``tensor`` is larger than ``clip_norm``, then
    tensor is rescaled to have norm equals to ``clip_norm``.

    This is wrapper function for ``tf.clip_by_norm``. See API documentation
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
    if isinstance(clip_norm, BaseWrapper):
        clip_norm = clip_norm.unwrap()

    _tensor = tf.clip_by_norm(
        tensor.unwrap(), clip_norm=clip_norm, axes=axes, name=name)
    return Tensor(tensor=_tensor, name=name)
