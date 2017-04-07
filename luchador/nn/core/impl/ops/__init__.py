"""Define operations"""
from __future__ import absolute_import
# pylint: disable=unused-import,redefined-builtin
from .clip import clip_by_value, clip_by_norm, clip_grads_by_norm  # noqa
from .grad import compute_gradient  # noqa
from .misc import build_sync_op, one_hot  # noqa
from .transform import reshape, tile  # noqa
from .math import (
    dot,
    abs, square, sqrt, exp, log, sin, cos,
    add, multiply, maximum, minimum,
    reduce_mean, reduce_sum, reduce_max,
)

__all__ = [
    'clip_by_value', 'clip_by_norm', 'clip_grads_by_norm',
    'add', 'multiply', 'maximum', 'minimum',
    'compute_gradient',
    'dot',
    'abs', 'square', 'sqrt',
    'exp', 'log', 'sin', 'cos',
    'reduce_mean', 'reduce_sum', 'reduce_max',
    'build_sync_op', 'one_hot',
    'reshape', 'tile',
]
