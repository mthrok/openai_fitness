from __future__ import absolute_import

import theano
import theano.tensor as T

from ..base import wrapper as base_wrapper

__all__ = ['Tensor', 'Input', 'Operation']

Operation = base_wrapper.Operation

_VARIABLES = {}


def _register_variable(name, var):
    if name in _VARIABLES:
        raise ValueError('Variable with name `{}` already exists.')
    _VARIABLES[name] = var


def retrieve_variable(name):
    return _VARIABLES.get(name)


class Variable(base_wrapper.BaseTensor):
    """Wrap SharedVariable object for storing network parameters"""
    def __init__(self, variable, name=None, trainable=True):
        """Wrap SharedVariable object.

        Args:
          variable (SharedVariable): theano.tensor.SharedVariable object
          name (str or None): When given, the name of the resulting wrapper is
            overwritten with this name, otherwise, name is constructed in the
            manner as Tensorflow.
        """
        name = name or variable.name
        val = variable.get_value()
        super(Variable, self).__init__(
            tensor=variable, shape=val.shape, name=name, dtype=val.dtype)
        _register_variable(name, self)
        self.trainable = trainable


class Tensor(base_wrapper.BaseTensor):
    """Wrap TensorVariable object for storing computation result"""
    def __init__(self, tensor, shape=None, name=None):
        """Wrap TensorVariable object.

        Args:
          tensor (TensorVariable): theano.tensor.TensorVariable object
          shape (list): Shape of the tensor being wrapped.
          name (str or None): Name of the resulting wrapper for convenience.
        """
        super(Tensor, self).__init__(
            tensor=tensor, shape=shape, name=name, dtype=tensor.dtype)


class Input(base_wrapper.BaseTensor):
    """Represents network input."""
    def __init__(self, shape, name=None, dtype=None):
        """Creates Input object which is converted to TensorVariable at build time

        Args:
          shape (list): The shape of the resulting object.
          name (str): The name of the resulting object.
          dtype (NumPy dtype or None): If None, default dtype(floatX) is used
        """
        super(Input, self).__init__(
            tensor=None, shape=shape, name=name, dtype=dtype)

    def __call__(self):
        return self.build()

    def build(self):
        dtype = self.dtype or theano.config.floatX
        if not self.shape:
            self.set(T.scalar(name=self.name, dtype=dtype))
            return self

        dim = len(self.shape)
        if dim == 1:
            t = T.vector(name=self.name, dtype=dtype)
        elif dim == 2:
            t = T.matrix(name=self.name, dtype=dtype)
        elif dim == 3:
            t = T.tensor3(name=self.name, dtype=dtype)
        elif dim == 4:
            t = T.tensor4(name=self.name, dtype=dtype)
        else:
            raise ValueError('shape length must be smaller than 5')
        self.set(t)
        return self
