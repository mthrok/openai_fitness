"""Define shape transformation operations"""
import numpy as np
import theano.tensor as T

from ..wrapper import Tensor

__all__ = ['reshape', 'tile']


def _infere_new_shape(original_shape, new_shape):
    if None in original_shape:
        return new_shape

    if -1 in new_shape:
        orig_size = np.prod(original_shape)
        known_size = np.abs(np.prod(new_shape))
        replace = orig_size // known_size
        return tuple(replace if s < 0 else s for s in new_shape)

    return new_shape


def reshape(var, new_shape, name=None):
    """Implement ``reshape`` in Theano backend.

    See :func:`luchador.nn.ops.reshape` for detail
    """
    _tensor = T.reshape(var.unwrap(), newshape=new_shape)
    new_shape = _infere_new_shape(var.shape, new_shape)
    return Tensor(tensor=_tensor, shape=new_shape, name=name)


def _compute_tile_shape(shape, pattern):
    if len(shape) > len(pattern):
        return _compute_tile_shape(pattern, shape)

    _shape = list(pattern)
    offset = len(pattern) - len(shape)
    for i, val in enumerate(shape):
        if _shape[offset + i] is None:
            continue
        if val is not None:
            _shape[offset + i] *= val
    return _shape


def tile(var, pattern, name=None):
    """Implement ``tile`` in Theano backend.

    See :func:`luchador.nn.ops.tile` for detail
    """
    _shape = _compute_tile_shape(pattern, var.shape)
    _tensor = T.tile(var.unwrap(), pattern)
    return Tensor(tensor=_tensor, shape=_shape, name=name)
