"""Module to define common interface for Tensor/Operation wrapping"""
from __future__ import absolute_import

import logging
from collections import OrderedDict

import numpy as np

__all__ = [
    'BaseWrapper', 'BaseTensor', 'BaseVariable', 'BaseInput', 'BaseOperation',
    'as_unwrapped'
]

_LG = logging.getLogger(__name__)


###############################################################################
# Mechanism for fetching Variable, Tensor, Input and Operation only by name
# This is similar to Tensorflow's get_variable function with reuse=Trie, but
# 1. Tensorflow's get_variable require dtype and shape for retrieving
# exisiting varaible, which is inconvenient.
# 2. It is extendef to Input, Tensor and Operation objects
_VARIABLES = OrderedDict()
_TENSORS = OrderedDict()
_INPUTS = OrderedDict()
_OPERATIONS = OrderedDict()


def register_variable(name, var):
    """Register variable to global list of variables for later resuse"""
    if name in _VARIABLES:
        raise ValueError('Variable `{}` already exists.'.format(name))
    _VARIABLES[name] = var


def register_tensor(name, tensor):
    """Register tensor to global list of tensors for later resuse"""
    if name in _TENSORS:
        _LG.warning('Tensor `%s` already exists.', name)
    _TENSORS[name] = tensor


def register_input(name, input_):
    """Register Input to global list of inputs for later resuse"""
    if name in _INPUTS:
        _LG.warning('Input `%s` already exists.', name)
    _INPUTS[name] = input_


def register_operation(name, operation):
    """Register Operation to global list of operations for later resuse"""
    if name in _OPERATIONS:
        _LG.warning('Operation `%s` already exists.', name)
    _OPERATIONS[name] = operation


def retrieve_variable(name):
    """Get variable from global list of variables"""
    if name not in _VARIABLES:
        raise ValueError('Variable `{}` does not exist.'.format(name))
    return _VARIABLES.get(name)


def retrieve_tensor(name):
    """Get tensor from global list of tensors

    Parameters
    ----------
    name : str
        Name of Tensor to fetch
    """
    if name not in _TENSORS:
        raise ValueError('Tensor `{}` does not exist.'.format(name))
    return _TENSORS.get(name)


def retrieve_input(name):
    """Get Input from global list of tensors

    Parameters
    ----------
    name : str
        Name of Input to fetch
    """
    if name not in _INPUTS:
        raise ValueError('Input `{}` does not exist.'.format(name))
    return _INPUTS[name]


def retrieve_operation(name):
    """Get Operation from global list of tensors

    Parameters
    ----------
    name : str
        Name of Operation to fetch
    """
    if name not in _OPERATIONS:
        raise ValueError('Operation `{}` does not exist.'.format(name))
    return _OPERATIONS[name]
###############################################################################


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
        self.dtype = dtype.name if isinstance(dtype, np.dtype) else dtype
        self.trainable = trainable

    def unwrap(self):
        """Get the underlying tensor object"""
        return self._tensor

    def set(self, obj):
        """Set the underlying tensor object"""
        self._tensor = obj

    def __repr__(self):
        return '<{}, {}, {}>'.format(self.name or '', self.dtype, self.shape)

    @property
    def size(self):
        """Return the number of elements in tensor"""
        if None in self.shape:
            return None
        return reduce(lambda x, y: x*y, self.shape, 1)

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


def as_unwrapped(value):
    """Unwrap if the given is wrapper type."""
    if isinstance(value, BaseWrapper):
        return value.unwrap()
    return value


class BaseVariable(BaseWrapper):
    """Base wrapper class for Variable"""
    def __init__(self, tensor, shape, name, dtype, trainable):
        super(BaseVariable, self).__init__(
            tensor=tensor, shape=shape, name=name,
            dtype=dtype, trainable=trainable)
        register_variable(name, self)


class BaseTensor(BaseWrapper):
    """Base wrapper class for Tensor"""
    def __init__(self, tensor, shape, name, dtype):
        super(BaseTensor, self).__init__(
            tensor=tensor, shape=shape, name=name, dtype=dtype)
        if name:
            self.register(name)

    def register(self, name):
        """Register this Tensor to global list of Tensors for later re-use

        Parameters
        ----------
        name : str
            Name to store this Tensor object.

        Note
        ----
        This method is provided for reusing Tensor which was created without
        a given name. Tensors created with proper name is automatically
        registered, thus there is no need to register.

        An exmaple of Tensor without name is those created with overloaded
        operator, such as __add__.

        >>> tensor2 = tensor1 + 0.1

        To retrieve `tensor2` with `get_tensor` method, you need to call
        tensor2.register().
        """
        register_tensor(name, self)


class BaseInput(BaseWrapper):
    """Base wrapper class for Input"""
    def __init__(self, tensor, shape, name, dtype):
        super(BaseInput, self).__init__(
            tensor=tensor, shape=shape, name=name, dtype=dtype)
        register_input(name, self)


class BaseOperation(object):
    """Wrapps theano updates or tensorflow operation"""
    def __init__(self, op, name=None):
        self.op = op
        self.name = name

        if name:
            register_operation(name, self)

    def unwrap(self):
        """Returns the underlying backend-specific operation object"""
        return self.op
