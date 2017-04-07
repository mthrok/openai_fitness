"""Implement elementwise math ops which work on single tensor"""
from __future__ import absolute_import

import theano.tensor as T

from ...wrapper import Tensor

__all__ = [
    'abs', 'square', 'sqrt', 'exp', 'log', 'sin', 'cos',
]
# pylint: disable=redefined-builtin,assignment-from-no-return


def abs(var, name=None):
    """Implement element-wise abs"""
    return var.__abs__(name=name)


def square(var, name=None):
    """Implement element-wise square"""
    _tensor = T.sqr(var.unwrap())
    return Tensor(tensor=_tensor, shape=var.shape, name=name)


def sqrt(var, name=None):
    """Implement element-wise sqrt"""
    _tensor = T.sqrt(var.unwrap())
    return Tensor(tensor=_tensor, shape=var.shape, name=name)


def exp(var, name=None):
    """Implement element-wise exp"""
    _tensor = T.exp(var.unwrap())
    return Tensor(tensor=_tensor, shape=var.shape, name=name)


def log(var, name=None):
    """Implement element-wise log"""
    _tensor = T.log(var.unwrap())
    return Tensor(tensor=_tensor, shape=var.shape, name=name)


def sin(var, name=None):
    """Implement element-wise sin"""
    _tensor = T.sin(var.unwrap())
    return Tensor(tensor=_tensor, shape=var.shape, name=name)


def cos(var, name=None):
    """Implement element-wise cos"""
    _tensor = T.cos(var.unwrap())
    return Tensor(tensor=_tensor, shape=var.shape, name=name)
