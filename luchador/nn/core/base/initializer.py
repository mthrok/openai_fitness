from __future__ import absolute_import

import logging

from luchador.common import get_subclasses, SerializeMixin

_LG = logging.getLogger(__name__)

__all__ = ['Initializer', 'get_initializer']


class Initializer(SerializeMixin, object):
    """Common interface for Initializer classes"""
    def __init__(self, **kwargs):
        """Validate and store arguments passed to subclass __init__ method

        As these **args are used to create a copy of instance, arguments which
        cannot be passed to constructor should not be passed. In other way, the
        signature of this constructor must match to the signature of the
        constractor of subclass object being created.
        """
        super(Initializer, self).__init__()
        self._store_args(**kwargs)

    ###########################################################################
    def __call__(self, shape):
        self.sample(shape)

    def sample(self, shape):
        """Sample random values for the given shape"""
        raise NotImplementedError(
            '`sample` method is not implemented for {}.{}'
            .format(type(self).__module__, type(self).__name__))


def get_initializer(name):
    """Get Initializer class with name"""
    for class_ in get_subclasses(Initializer):
        if class_.__name__ == name:
            return class_
    raise ValueError('Unknown Initializer: {}'.format(name))
