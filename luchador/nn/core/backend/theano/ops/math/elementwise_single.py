"""Define elementwise math ops which work on single tensor"""
from __future__ import absolute_import

import theano.tensor as T

from ...wrapper import Tensor

__all__ = [
    'abs', 'square', 'sqrt', 'exp', 'log', 'sin', 'cos',
]
# pylint: disable=redefined-builtin,assignment-from-no-return


def abs(var, name=None):
    """Element-wise absolute value"""
    return var.__abs__(name=name)


def square(var, name=None):
    """Returns square of the given variable"""
    _tensor = T.sqr(var.unwrap())
    return Tensor(tensor=_tensor, shape=var.shape, name=name)


def sqrt(var, name=None):
    """Returns square root of the given variable"""
    _tensor = T.sqrt(var.unwrap())
    return Tensor(tensor=_tensor, shape=var.shape, name=name)


def exp(var, name=None):
    """Returns exponential of the given variable"""
    _tensor = T.exp(var.unwrap())
    return Tensor(tensor=_tensor, shape=var.shape, name=name)


def log(var, name=None):
    """Returns exponential of the given variable"""
    _tensor = T.log(var.unwrap())
    return Tensor(tensor=_tensor, shape=var.shape, name=name)


def sin(var, name=None):
    """Returns sin of the given variable"""
    _tensor = T.sin(var.unwrap())
    return Tensor(tensor=_tensor, shape=var.shape, name=name)


def cos(var, name=None):
    """Returns cos of the given variable"""
    _tensor = T.cos(var.unwrap())
    return Tensor(tensor=_tensor, shape=var.shape, name=name)
