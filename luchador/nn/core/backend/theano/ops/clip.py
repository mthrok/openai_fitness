"""Define clipping methods"""
import theano.tensor as T

from luchador.nn.core.base import BaseWrapper
from ..wrapper import Tensor

__all__ = ['clip_by_value', 'clip_by_norm']


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
    _tensor = tensor.unwrap().clip(a_max=max_value, a_min=min_value)
    return Tensor(tensor=_tensor, shape=tensor.shape, name=name)


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
    shape = tensor.shape
    tensor = tensor.unwrap()
    if _is_wrapper(clip_norm):
        clip_norm = clip_norm.unwrap()
    # Ideally we want do this without unwrapping the variables but
    # `minimum` cannot handle broadcasting yet
    clip_norm_i = 1.0 / clip_norm
    l2norm_i = 1.0 / T.sqrt((tensor * tensor).sum(axis=axes, keepdims=True))
    _tensor = tensor * clip_norm * T.minimum(l2norm_i, clip_norm_i)
    return Tensor(tensor=_tensor, shape=shape, name=name)
