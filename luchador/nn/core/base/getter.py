"""Define common interface for Cost classes"""
from __future__ import absolute_import

from luchador.util import get_subclasses

from ..base import BaseCost, BaseLayer, BaseOptimizer, BaseInitializer


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
