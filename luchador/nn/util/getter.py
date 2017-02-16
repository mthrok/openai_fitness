"""Define common interface for Cost classes"""
from __future__ import absolute_import

from luchador.util import get_subclasses
import luchador.nn
from ..base import BaseCost, BaseLayer, BaseOptimizer, BaseInitializer
from ..base import wrapper as base_wrapper
from ..model import BaseModel


###############################################################################
# Functions for fetching class
def get_cost(name):
    """Get ``Cost`` class by name

    Parameters
    ----------
    name : str
        Type of ``Cost`` to get.

    Returns
    -------
    type
        ``Cost`` type found

    Raises
    ------
    ValueError
        When ``Cost`` class with the given type is not found
    """
    for class_ in get_subclasses(BaseCost):
        if class_.__name__ == name:
            return class_
    raise ValueError('Unknown Cost: {}'.format(name))


def get_layer(name):
    """Get ``Layer`` class by name

    Parameters
    ----------
    name : str
        Type of ``Layer`` to get

    Returns
    -------
    type
        ``Layer`` type found

    Raises
    ------
    ValueError
        When ``Layer`` class with the given type is not found
    """
    for class_ in get_subclasses(BaseLayer):
        if class_.__name__ == name:
            return class_
    raise ValueError('Unknown Layer: {}'.format(name))


def get_optimizer(name):
    """Get ``Optimizer`` class by name

    Parameters
    ----------
    name : str
        Type of ``Optimizer`` to get

    Returns
    -------
    type
        ``Optimizer`` type found

    Raises
    ------
    ValueError
        When ``Optimizer`` class with the given type is not found
    """
    for class_ in get_subclasses(BaseOptimizer):
        if class_.__name__ == name:
            return class_
    raise ValueError('Unknown Optimizer: {}'.format(name))


def get_initializer(name):
    """Get ``Initializer`` class by name

    Parameters
    ----------
    name : str
        Type of ``Initializer`` to get

    Returns
    -------
    type
        ``Initializer`` type found

    Raises
    ------
    ValueError
        When ``Initializer`` class with the given type is not found
    """
    for class_ in get_subclasses(BaseInitializer):
        if class_.__name__ == name:
            return class_
    raise ValueError('Unknown Initializer: {}'.format(name))


# -----------------------------------------------------------------------------
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
    for class_ in luchador.util.get_subclasses(BaseModel):
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
        scope = luchador.nn.backend.get_variable_scope().name
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
        scope = luchador.nn.backend.get_variable_scope().name
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
        scope = luchador.nn.backend.get_variable_scope().name
        return base_wrapper.retrieve_operation('{}/{}'.format(scope, name))
    except ValueError:
        pass
    return base_wrapper.retrieve_operation(name)
