"""Module for defining input variable/tensor/input wrapper"""
from __future__ import absolute_import

import numbers

import tensorflow as tf

import luchador
from luchador.nn.base import wrapper as base_wrapper
from luchador.nn.base.wrapper import Operation

__all__ = ['Variable', 'Tensor', 'Input', 'Operation']


###############################################################################
# Mechanism for enabling reusing variable without explicitly giving dtype or
# shape. When creating Variable with get_variable and reuse=False, we store
# mapping from name to the resulting Variable wrapper.
# When retrieving a Variable under reuse=True, we return the stored variable.
_VARIABLES = {}


def _register_variable(name, var):
    if name in _VARIABLES:
        raise ValueError('Variable `{}` already exists.'.format(name))
    _VARIABLES[name] = var


def retrieve_variable(name):
    """Get variable from global list of variables"""
    return _VARIABLES.get(name)
###############################################################################


class Variable(base_wrapper.BaseTensor):
    """Wrap tf.Variable object for storing network parameters"""
    def __init__(self, variable, name=None, trainable=True):
        """Wrap Tensorflow Variable object.

        Args:
          variable (tf.Variable): Tensorflow Variable object
          name (str or None): When given, the name of the resulting wrapper is
            overwritten with this name.
        """
        name = name or variable.op.name
        shape = variable.get_shape().as_list()
        dtype = variable.dtype.as_numpy_dtype
        super(Variable, self).__init__(
            tensor=variable, shape=shape, name=name, dtype=dtype)
        _register_variable(name, self)
        self.trainable = trainable


class Tensor(base_wrapper.BaseTensor):
    """Wrap tf.Tensor object for storing computation result"""
    def __init__(self, tensor, shape=None, name=None):
        """Wrap Tensorflow Tensor object.

        When wrapping Tensor object, as shape and name can be retrieved from
        the object being wrapped, you need not to give them explicitly. You can
        overwrite name attribute for some reasons by giving one.
        The shape argument is here for compatibility with Theano backend and
        not used in tensorflow backend.

        Args:
          tensor (tf.Tensor): Tensorflow Tensor object.
          shape : Not used.
          name (str or None): When given, the name of the resulting wrapper is
            overwritten with this name.
        """
        name = name or tensor.name
        shape = tensor.get_shape().as_list()
        dtype = tensor.dtype.as_numpy_dtype
        super(Tensor, self).__init__(
            tensor=tensor, shape=shape, name=name, dtype=dtype)

    def __mul__(self, other):
        """Scalar multiplication"""
        if not isinstance(other, numbers.Number):
            return NotImplemented

        return Tensor(tensor=self._tensor * other)

    def __rmul__(self, other):
        return self * other

    def __add__(self, other):
        if isinstance(other, numbers.Number):
            pass
        elif not self.shape == other.shape:
            raise ValueError(
                'Inconsistent shape: {} and {}'
                .format(self.shape, other.shape)
            )
        else:
            other = other.unwrap()

        return Tensor(tensor=self._tensor + other)

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return Tensor(tensor=-self._tensor)


class Input(base_wrapper.BaseTensor):
    """Represents network input."""
    def __init__(self, shape, name=None, dtype=None):
        """Creates Input object which is converted to placeholder at build time

        Args:
          shape (list): The shape of the resulting object.
          name (str): The name of the resulting object.
          dtype (NumPy dtype or None): If None, default dtype is used
        """
        super(Input, self).__init__(
            tensor=None, shape=shape, name=name, dtype=dtype)

    def __call__(self):
        return self.build()

    def build(self):
        """Instantiate underlying Variable"""
        if self.unwrap() is None:
            dtype = self.dtype or luchador.get_nn_dtype()
            pf = tf.placeholder(dtype=dtype, shape=self.shape, name=self.name)
            self.set(pf)
        return self
