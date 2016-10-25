from __future__ import absolute_import

import abc
import logging

from luchador import common

_LG = logging.getLogger(__name__)


class BaseInitializer(common.SerializeMixin, object):
    """Define Common interface for Initializer classes"""
    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwargs):
        super(BaseInitializer, self).__init__()
        self._store_args(**kwargs)

    def sample(self, shape):
        """Sample random values in the given shape

        Parameters
        ----------
        shape : tuple
            shape of array to sample

        Returns
        -------
        [Theano backend] : Numpy Array
            Sampled value.
        [Tensorflow backend] : None
            In Tensorflow backend, sampling is handled by underlying native
            Initializers and this method is not used.
        """
        return self._sample(shape)

    @abc.abstractmethod
    def _sample(self, shape):
        pass


def get_initializer(name):
    """Retrieve Initializer class by name

    Parameters
    ----------
    name : str
        Name of Initializer to retrieve

    Returns
    -------
    type
        Initializer type found

    Raises
    ------
    ValueError
        When Initializer with the given name is not found
    """
    for class_ in common.get_subclasses(BaseInitializer):
        if class_.__name__ == name:
            return class_
    raise ValueError('Unknown Initializer: {}'.format(name))
