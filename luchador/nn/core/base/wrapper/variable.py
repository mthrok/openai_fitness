"""Define base Variable type"""
from __future__ import absolute_import

from ..scope import get_variable_scope
from .store import register, retrieve
from .wrapper import BaseWrapper

__all__ = ['BaseVariable', 'get_variable']


class BaseVariable(BaseWrapper):
    """Base wrapper class for Variable"""
    def __init__(self, tensor, shape, name, dtype, trainable):
        super(BaseVariable, self).__init__(
            tensor=tensor, shape=shape, name=name,
            dtype=dtype, trainable=trainable)
        register('variable', name, self)


def get_variable(name):
    """Get an instance of ``Variable`` from the current or the global scope

    Parameters
    ----------
    name : str
        name of ``Variable`` instance to get

    Returns
    -------
    Variable
    """
    scope = get_variable_scope().name
    try:
        name_ = '{}/{}'.format(scope, name) if scope else name
        return retrieve('variable', name_)
    except ValueError:
        pass
    return retrieve('variable', name)
