"""Define shape transformation operations"""
import numpy as np
import theano.tensor as T

import luchador.util
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
    """Reshape tensor.

    Parameters
    ----------
    new_shape : tuple
        new shape

    name : str
        Name of operation

    Returns
    -------
    Tensor
        Tensor with new shape

    Notes
    -----
    This function is for conveniently invoke underlying reshap function.
    Shape-checking and inference is not carried out.
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
    """Tile tensor.

    Parameters
    ----------
    pattern : tuple
        tile pattern

    name : str
        Name of operation

    Returns
    -------
    Tensor
        Resulting tensor.

    Notes
    -----
    Currently only constant pattern is allowed.
    """
    if not luchador.util.is_iteratable(pattern):
        raise ValueError('`pattern` must be iteratable')

    _shape = _compute_tile_shape(pattern, var.shape)
    _tensor = T.tile(var.unwrap(), pattern)
    return Tensor(tensor=_tensor, shape=_shape, name=name)
