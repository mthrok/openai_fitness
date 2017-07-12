"""Define common Wrapper interface"""
from __future__ import absolute_import

__all__ = ['BaseWrapper']


def _parse_dtype(dtype):
    try:
        return dtype.name
    except AttributeError:
        return dtype


def _product(shape):
    ret = 1
    for elem in shape:
        ret *= elem
    return ret


class BaseWrapper(object):
    """Wraps Tensor or Variable object in Theano/Tensorflow

    This class was introduced to provide easy shape inference to Theano Tensors
    while having the common interface for both Theano and Tensorflow.
    `unwrap` method provides access to the underlying object.
    """
    def __init__(self, tensor, shape, name, dtype, trainable=False):
        self._tensor = tensor
        self.shape = tuple(shape)
        self.name = name
        self.dtype = _parse_dtype(dtype)
        self.trainable = trainable

    def unwrap(self):
        """Get the underlying tensor object"""
        return self._tensor

    def set(self, obj):
        """Set the underlying tensor object"""
        self._tensor = obj

    def __repr__(self):
        class_name = self.__class__.__name__.split('.')[-1]
        wrapper_name = self.name or 'No Name'
        return '<{}:{}, {}, {}: {}>'.format(
            class_name, id(self._tensor), self.dtype, self.shape, wrapper_name)

    @property
    def size(self):
        """Return the number of elements in tensor"""
        if None in self.shape:
            return None
        return _product(self.shape)

    @property
    def n_dim(self):
        """Return the number of array dimension in tensor"""
        return len(self.shape)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __div__(self, other):
        return self.__truediv__(other)

    def __truediv__(self, other):
        return NotImplemented

    def __rdiv__(self, other):
        return self.__rtruediv__(other)

    def __rtruediv__(self, other):
        return NotImplemented
