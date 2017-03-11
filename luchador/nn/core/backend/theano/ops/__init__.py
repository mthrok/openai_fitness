"""Define operations over Tensor wrappers"""
from __future__ import absolute_import
# pylint: disable=redefined-builtin
from .clip import clip_by_value, clip_by_norm
from .elementwise import add, multiply, maximum, minimum
from .grad import compute_gradient
from .math import (
    abs, square, sqrt, exp, log, sin, cos,
    reduce_mean, reduce_sum, reduce_max,
)
from .misc import build_sync_op, one_hot
from .transform import reshape, tile

__all__ = [
    'clip_by_value', 'clip_by_norm',
    'add', 'multiply', 'maximum', 'minimum',
    'compute_gradient',
    'abs', 'square', 'sqrt',
    'exp', 'log', 'sin', 'cos',
    'reduce_mean', 'reduce_sum', 'reduce_max',
    'build_sync_op', 'one_hot',
    'reshape', 'tile',
]
