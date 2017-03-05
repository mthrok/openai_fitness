"""Define interface of Random source"""
from __future__ import absolute_import

import abc

__all__ = ['BaseRandomSource']


class BaseRandomSource(object):
    """Create Tensor which represents random value"""
    __metaclass__ = abc.ABCMeta

    def sample(self, shape, dtype):
        """Sample uniform random value from distribution

        Parameters
        ----------
        shape : tuple
            Shape of sample
        dtype : str
            data type of sample
        """
        return self._sample(shape=shape, dtype=dtype)

    @abc.abstractmethod
    def _sample(self, shape, dtype):
        pass
