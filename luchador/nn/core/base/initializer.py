from __future__ import absolute_import

import abc
import logging

from luchador import common

_LG = logging.getLogger(__name__)


class BaseInitializer(common.SerializeMixin, object):
    """Common interface for Initializer classes"""
    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwargs):
        """Validate and store arguments passed to subclass __init__ method

        As these **args are used to create a copy of instance, arguments which
        cannot be passed to constructor should not be passed. In other way, the
        signature of this constructor must match to the signature of the
        constractor of subclass object being created.
        """
        super(BaseInitializer, self).__init__()
        self._store_args(**kwargs)

    @abc.abstractmethod
    def sample(self, shape):
        """Sample random values for the given shape"""
        pass


def get_initializer(name):
    """Get Initializer class with name"""
    for class_ in common.get_subclasses(BaseInitializer):
        if class_.__name__ == name:
            return class_
    raise ValueError('Unknown Initializer: {}'.format(name))
