"""Define common interface for Cost classes"""
from __future__ import absolute_import

from luchador.util import get_subclasses

from ..model import BaseModel

__all__ = ['get_model']


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
