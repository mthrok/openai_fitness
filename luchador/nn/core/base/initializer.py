"""Define base initializer class"""
from __future__ import absolute_import

import abc
import logging

import luchador.util

__all__ = ['BaseInitializer', 'fetch_initializer']
_LG = logging.getLogger(__name__)


class BaseInitializer(luchador.util.StoreMixin, object):
    """Define Common interface for Initializer classes"""
    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwargs):
        super(BaseInitializer, self).__init__()
        self._store_args(**kwargs)

        # Backend-specific initialization is run here
        self._run_backend_specific_init()

    @abc.abstractmethod
    def _run_backend_specific_init(self):
        """Backend-specific initilization"""
        pass

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


def fetch_initializer(name):
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
    for class_ in luchador.util.fetch_subclasses(BaseInitializer):
        if class_.__name__ == name:
            return class_
    raise ValueError('Unknown Initializer: {}'.format(name))
