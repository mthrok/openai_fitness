"""Implement elementwise math ops which work on single tensor"""
from __future__ import absolute_import

from .... import backend as be

__all__ = [
    'abs', 'square', 'sqrt', 'exp', 'log', 'sin', 'cos',
]
# pylint: disable=redefined-builtin


def abs(var, name=None):
    """Compute element-wise absolute value"""
    return be.ops.abs(var, name=name)


def square(var, name=None):
    """Compute element-wise square"""
    return be.ops.square(var, name=name)


def sqrt(var, name=None):
    """Compute element-wise squared root"""
    return be.ops.sqrt(var, name=name)


def exp(var, name=None):
    """Compute element-wise exponential"""
    return be.ops.exp(var, name=name)


def log(var, name=None):
    """Compute element-wise logarithm"""
    return be.ops.log(var, name=name)


def sin(var, name=None):
    """Compute element-wise sin"""
    return be.ops.sin(var, name=name)


def cos(var, name=None):
    """Compute element-wise cos"""
    return be.ops.cos(var, name=name)
