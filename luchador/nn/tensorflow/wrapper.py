"""Module for defining input variable/tensor/input wrapper"""
from __future__ import absolute_import

import numbers

import tensorflow as tf

import luchador
import luchador.util
from luchador.nn.base import wrapper as base_wrapper

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


class TensorMixin(object):  # pylint: disable=too-few-public-methods
    """Add elementwise operations to Tensor class"""
    def _extract_operand(self, other):
        if isinstance(other, numbers.Number):
            return other
        if self.shape == other.shape:
            return other.unwrap()
        raise ValueError(
            'Inconsistent shape: {} and {}'.format(self.shape, other.shape)
        )

    def __neg__(self):
        return Tensor(tensor=-self._tensor)

    def __add__(self, other):
        _other = self._extract_operand(other)
        return Tensor(tensor=self._tensor + _other)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        """Scalar subtraction or elementwise subtraction"""
        _other = self._extract_operand(other)
        return Tensor(tensor=self._tensor-_other)

    def __rsub__(self, other):
        _other = self._extract_operand(other)
        return Tensor(tensor=_other-self._tensor)

    def __mul__(self, other):
        """Scalar multiplication"""
        _other = self._extract_operand(other)
        return Tensor(tensor=self._tensor * _other)

    def __rmul__(self, other):
        return self * other

    def mean(self, axis=None, keep_dims=False, name=None):
        """Compute mean across the given axis

        Parameters
        ----------
        axis : int, list or None
            The dimensions to reduce. If None (the default),
            reduces all dimensions.
        keep_dims: bool
            If true, retains reduced dimensions with length 1.
        name: str
            A name for the operation.
        """
        _tensor = tf.reduce_mean(
            self._tensor, axis=axis, keep_dims=keep_dims, name=name)
        return Tensor(tensor=_tensor, name=name)


class Variable(TensorMixin, base_wrapper.BaseTensor):
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


class Tensor(TensorMixin, base_wrapper.BaseTensor):
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


class Input(TensorMixin, base_wrapper.BaseTensor):
    """Represents network input."""
    def __init__(self, shape, name=None, dtype=None):
        """Creates Input object which wraps placeholder

        Args:
          shape (list): The shape of the resulting object.
          name (str): The name of the resulting object.
          dtype (NumPy dtype or None): If None, default dtype is used
        """
        _dtype = dtype or luchador.get_nn_dtype()
        tensor = tf.placeholder(dtype=_dtype, shape=shape, name=name)
        super(Input, self).__init__(
            tensor=tensor, shape=shape, name=name, dtype=dtype)


class Operation(base_wrapper.Operation):
    """Wrap tensorflow operations"""
    def __init__(self, op, name=None):
        if luchador.util.is_iteratable(op):
            op = tf.group(*op, name=name)

        super(Operation, self).__init__(op=op, name=name)
