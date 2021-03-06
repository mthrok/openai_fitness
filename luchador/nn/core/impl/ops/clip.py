"""Define clipping methods"""
from __future__ import absolute_import

from luchador.nn.core.base import BaseWrapper
from ... import backend as be

__all__ = ['clip_by_value', 'clip_by_norm', 'clip_grads_by_norm']


def _is_wrapper(obj):
    return isinstance(obj, BaseWrapper)


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
    if not _is_wrapper(max_value) and not _is_wrapper(min_value):
        if max_value < min_value:
            raise ValueError('`max_value` must be larger than `min_value`')
    if _is_wrapper(max_value):
        max_value = max_value.unwrap()
    if _is_wrapper(min_value):
        min_value = min_value.unwrap()
    return be.ops.clip_by_value(tensor, max_value, min_value, name=name)


def clip_by_norm(tensor, clip_norm, axes=None, name=None):
    """Clip tensor values to a maximum L2-norm.

    If the norm of the input ``tensor`` is larger than ``clip_norm``, then
    tensor is rescaled to have norm equals to ``clip_norm``.

    This function is mimic for ``tf.clip_by_norm``. See API documentation
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
    if _is_wrapper(clip_norm):
        clip_norm = clip_norm.unwrap()
    return be.ops.clip_by_norm(tensor, clip_norm, axes, name)


def clip_grads_by_norm(grads_and_vars, clip_norm):
    """Clip each gradient by norm

    Parameters
    ----------
    grads_and_vars : list
        Gradient and Variable tuples. Return value from
        :func:`luchador.nn.ops.compute_gradients`.

    clip_norm : Number or Tensor
        Value to clip gradients

    Returns
    -------
    list
        Resulting gradients and vars pairs
    """
    ret = []
    for grad, var in grads_and_vars:
        name = '{}_clip'.format(grad.name)
        grad = clip_by_norm(grad, clip_norm=clip_norm, name=name)
        ret.append((grad, var))
    return ret
