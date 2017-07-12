"""Define base Input type"""
from __future__ import absolute_import

from ..scope import get_variable_scope
from .store import register, retrieve
from .wrapper import BaseWrapper

__all__ = ['BaseInput', 'get_input']


class BaseInput(BaseWrapper):
    """Base wrapper class for Input"""
    def __init__(self, tensor, shape, name, dtype):
        super(BaseInput, self).__init__(
            tensor=tensor, shape=shape, name=name, dtype=dtype)
        if name:
            register('input', name, self)


def get_input(name):
    """Get an instance of ``Input`` from the current or the global scope

    Parameters
    ----------
    name : str
        name of ``Input`` instance to get

    Returns
    -------
    Input
    """
    try:
        scope = get_variable_scope().name
        name_ = '{}/{}'.format(scope, name) if scope else name
        return retrieve('input', name_)
    except ValueError:
        pass
    return retrieve('input', name)
