"""Module to define common interface for Tensor/Operation wrapping"""
from __future__ import absolute_import

import abc
import logging
from collections import OrderedDict

import numpy as np

from . import scope as scope_module

__all__ = [
    'BaseRandomSource', 'BaseWrapper', 'BaseTensor', 'BaseVariable',
    'BaseInput', 'BaseOperation',
    'as_unwrapped',
    'get_input', 'get_variable', 'get_tensor', 'get_operation', 'get_grad',
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


def _register_variable(name, var):
    if name in _VARIABLES:
        raise ValueError('Variable `{}` already exists.'.format(name))
    _VARIABLES[name] = var


def _register_tensor(name, tensor):
    if name in _TENSORS:
        _LG.warning('Tensor `%s` already exists.', name)
    _TENSORS[name] = tensor


def _register_input(name, input_):
    if name in _INPUTS:
        _LG.warning('Input `%s` already exists.', name)
    _INPUTS[name] = input_


def _register_operation(name, operation):
    if name in _OPERATIONS:
        _LG.warning('Operation `%s` already exists.', name)
    _OPERATIONS[name] = operation


def _retrieve_variable(name):
    if name not in _VARIABLES:
        raise ValueError('Variable `{}` does not exist.'.format(name))
    return _VARIABLES.get(name)


def _retrieve_tensor(name):
    if name not in _TENSORS:
        raise ValueError('Tensor `{}` does not exist.'.format(name))
    return _TENSORS.get(name)


def _retrieve_input(name):
    if name not in _INPUTS:
        raise ValueError('Input `{}` does not exist.'.format(name))
    return _INPUTS[name]


def _retrieve_operation(name):
    if name not in _OPERATIONS:
        raise ValueError('Operation `{}` does not exist.'.format(name))
    return _OPERATIONS[name]
###############################################################################


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
        _register_variable(name, self)


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
        _register_tensor(name, self)


class BaseInput(BaseWrapper):
    """Base wrapper class for Input"""
    def __init__(self, tensor, shape, name, dtype):
        super(BaseInput, self).__init__(
            tensor=tensor, shape=shape, name=name, dtype=dtype)
        _register_input(name, self)


class BaseOperation(object):
    """Wrapps theano updates or tensorflow operation"""
    def __init__(self, op, name=None):
        self.op = op
        self.name = name

        if name:
            _register_operation(name, self)

    def unwrap(self):
        """Returns the underlying backend-specific operation object"""
        return self.op


###############################################################################
def get_variable(name):
    """Get an instance of ``Variable`` from the current or the global scope

    Parameters
    ----------
    name : str
        name of ``Variable`` instance to get

    Returns
    -------
    Variable
    """
    scope = scope_module.get_variable_scope().name
    try:
        name_ = '{}/{}'.format(scope, name) if scope else name
        return _retrieve_variable(name_)
    except ValueError:
        pass
    return _retrieve_variable(name)


def get_input(name):
    """Get an instance of ``Input`` from the current or the global scope

    Parameters
    ----------
    name : str
        name of ``Input`` instance to get

    Returns
    -------
    Input
    """
    try:
        scope = scope_module.get_variable_scope().name
        name_ = '{}/{}'.format(scope, name) if scope else name
        return _retrieve_input(name_)
    except ValueError:
        pass
    return _retrieve_input(name)


def get_tensor(name):
    """Get an instance of ``Tensor`` from the current or the global scope

    Parameters
    ----------
    name : str
        name of ``Tensor`` instance to get

    Returns
    -------
    Tensor
    """
    try:
        scope = scope_module.get_variable_scope().name
        name_ = '{}/{}'.format(scope, name) if scope else name
        return _retrieve_tensor(name_)
    except ValueError:
        pass
    return _retrieve_tensor(name)


def get_grad(var):
    """Get gradient ``Tensor`` corresponding to the given ``Variable``

    In optimizers, gradient tensors are registered in global scope,
    following the naming pattern ``<scope>/<variable_name>_grad``.

    This function automatically build such name from the given ``Variable``
    and the current scope name.

    To properly fetch the corresponding gradient ``Tensor``, this function
    must be called in the scope where gradient ``Tensor`` was defined.

    Examples
    --------
    >>> from luchador import nn
    >>> x = nn.get_variable(shape=(), name='x')
    >>> # Variable x is registered with name 'x'
    >>> y = x * x
    >>> sgd = nn.optimizer.SGD(learning_rate=0.1)
    >>> with nn.variable_scope('optimization'):
    >>>    sgd.minimize(loss=y, wrt=x)
    >>>    # dydx is registered with name '/optimization/x_grad'
    >>>    dydx2 = nn.get_grad_tensor(x)
    >>>    assert dydx1 is dydx2

    Parameters
    ----------
    var : Variable
        ``Variable`` object of which grad is retrieved.

    Returns
    -------
    Tensor
        ``Tensor`` object which is a gradient of given ``Variable``
    """
    return get_tensor('{}_grad'.format(var.name))


def get_operation(name):
    """Get ``Operation`` instance from the current scope or the global scope

    Parameters
    ----------
    name : str
        name of ``Operation`` instance to get

    Returns
    -------
    Operation
    """
    try:
        scope = scope_module.get_variable_scope().name
        name_ = '{}/{}'.format(scope, name) if scope else name
        return _retrieve_operation(name_)
    except ValueError:
        pass
    return _retrieve_operation(name)
