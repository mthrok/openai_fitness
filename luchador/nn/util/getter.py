"""Define common interface for Cost classes"""
from __future__ import absolute_import

from luchador.util import get_subclasses

from .. import core
from ..core.base import wrapper as base_wrapper
from ..core.base.getter import (
    get_cost, get_layer, get_optimizer, get_initializer
)
from ..model import BaseModel

__all__ = [
    'get_cost', 'get_layer', 'get_optimizer', 'get_initializer',
    'get_model', 'get_input', 'get_tensor', 'get_grad', 'get_operation',
]


def get_model(name):
    """Get ``Model`` class by name

    Parameters
    ----------
    name : str
        Name of ``Model`` to get

    Returns
    -------
    type
        ``Model`` type found

    Raises
    ------
    ValueError
        When ``Model`` class with the given name is not found
    """
    for class_ in get_subclasses(BaseModel):
        if class_.__name__ == name:
            return class_
    raise ValueError('Unknown model: {}'.format(name))


###############################################################################
# Functions for fetching instance
def get_input(name):
    """Get ``Input`` instance from the current scope or the global scope

    Parameters
    ----------
    name : str
        name of ``Input`` instance to get

    Returns
    -------
    Input
    """
    try:
        scope = core.get_variable_scope().name
        return base_wrapper.retrieve_input('{}/{}'.format(scope, name))
    except ValueError:
        pass
    return base_wrapper.retrieve_input(name)


def get_tensor(name):
    """Get ``Tensor`` instance from the current scope or the global scope

    Parameters
    ----------
    name : str
        name of ``Tensor`` instance to get

    Returns
    -------
    Tensor
    """
    try:
        scope = core.get_variable_scope().name
        return base_wrapper.retrieve_tensor('{}/{}'.format(scope, name))
    except ValueError:
        pass
    return base_wrapper.retrieve_tensor(name)


def get_grad(var):
    """Get gradient ``Tensor`` corresponding to the given ``Variable``

    In optimizers, gradient tensors are registered in global scope,
    following the naming pattern ``<scope>/<variable_name>_grad``.

    This function automatically build such name from the given ``Variable``
    and the current scope name.

    To properly fetch the corresponding gradient ``Tensor``, this function
    must be called in the scope where gradient ``Tensor`` was defined.

    Examples
    --------
    >>> from luchador import nn
    >>> x = nn.get_variable(shape=(), name='x')
    >>> # Variable x is registered with name 'x'
    >>> y = x * x
    >>> sgd = nn.optimizer.SGD(learning_rate=0.1)
    >>> with nn.variable_scope('optimization'):
    >>>    sgd.minimize(loss=y, wrt=x)
    >>>    # dydx is registered with name '/optimization/x_grad'
    >>>    dydx2 = nn.get_grad_tensor(x)
    >>>    assert dydx1 is dydx2

    Parameters
    ----------
    var : Variable
        ``Variable`` object of which grad is retrieved.

    Returns
    -------
    Tensor
        ``Tensor`` object which is a gradient of given ``Variable``
    """
    return get_tensor('{}_grad'.format(var.name))


def get_operation(name):
    """Get ``Operation`` instance from the current scope or the global scope

    Parameters
    ----------
    name : str
        name of ``Operation`` instance to get

    Returns
    -------
    Operation
    """
    try:
        scope = core.get_variable_scope().name
        return base_wrapper.retrieve_operation('{}/{}'.format(scope, name))
    except ValueError:
        pass
    return base_wrapper.retrieve_operation(name)
