"""Define base network model structure and fetch method"""
from __future__ import absolute_import

from luchador.util import get_subclasses

__all__ = ['BaseModel', 'get_model']


class BaseModel(object):  # pylint: disable=too-few-public-methods
    """Base Model class"""
    def __init__(self):
        super(BaseModel, self).__init__()


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
