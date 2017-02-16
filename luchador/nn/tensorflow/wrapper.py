"""Module for defining input variable/tensor/input wrapper"""
from __future__ import division
from __future__ import absolute_import

import numbers

import tensorflow as tf

import luchador
import luchador.util
from luchador.nn.base import wrapper as base_wrapper
from . import misc

__all__ = ['Variable', 'Tensor', 'Input', 'Operation']


class TensorMixin(object):  # pylint: disable=too-few-public-methods
    """Add elementwise operations to Tensor class"""
    def _extract_operand(self, other):
        """Extract operand for elementwise operation"""
        if isinstance(other, numbers.Number):
            return other
        if self.shape == other.shape:
            return other.unwrap()
        if self.size == 1 or other.size == 1:
            return other.unwrap()
        raise ValueError(
            'Inconsistent shape: {} and {}'.format(self.shape, other.shape)
        )

    def __neg__(self, name=None):
        return type(self)(tensor=-self._tensor, shape=self.shape, name=name)

    def __add__(self, other, name=None):
        _other = self._extract_operand(other)
        return Tensor(tensor=self._tensor + _other, name=name)

    def __sub__(self, other, name=None):
        """Scalar subtraction or elementwise subtraction"""
        _other = self._extract_operand(other)
        return Tensor(tensor=self._tensor - _other, name=name)

    def __rsub__(self, other, name=None):
        _other = self._extract_operand(other)
        return Tensor(tensor=_other - self._tensor, name=name)

    def __mul__(self, other, name=None):
        """Scalar multiplication"""
        _other = self._extract_operand(other)
        return Tensor(tensor=self._tensor * _other, name=name)

    def __truediv__(self, other, name=None):
        _other = self._extract_operand(other)
        return Tensor(tensor=self._tensor / _other, name=name)

    def __rtruediv__(self, other, name=None):
        _other = self._extract_operand(other)
        return Tensor(tensor=_other / self._tensor, name=name)

    def __floordiv__(self, other, name=None):
        _other = self._extract_operand(other)
        return Tensor(tensor=self._tensor//_other, name=name)

    def __rfloordiv__(self, other, name=None):
        _other = self._extract_operand(other)
        return Tensor(tensor=_other//self._tensor, name=name)

    def mean(self, axis=None, keep_dims=False, name=None):
        """:any:`luchador.nn.tesorflow.misc.mean`"""
        return misc.mean(self, axis=axis, keep_dims=keep_dims, name=name)

    def sum(self, axis=None, keep_dims=False, name=None):
        """:any:`luchador.nn.tesorflow.misc.sum`"""
        return misc.sum(self, axis=axis, keep_dims=keep_dims, name=name)

    def max(self, axis=None, keep_dims=False, name=None):
        """:any:`luchador.nn.tesorflow.misc.max`"""
        return misc.max(self, axis=axis, keep_dims=keep_dims, name=name)

    def clip(self, max_value, min_value, name=None):
        """:any:`luchador.nn.tesorflow.misc.clip`"""
        return misc.clip(
            self, max_value=max_value, min_value=min_value, name=name)

    def reshape(self, new_shape, name=None):
        """:any:`luchador.nn.tesorflow.misc.clip`"""
        return misc.reshape(self, new_shape=new_shape, name=name)

    def tile(self, pattern, name=None):
        """:any:`luchador.nn.tesorflow.misc.clip`"""
        return misc.reshape(self, pattern=pattern, name=name)


def _get_dtype_str(tensor):
    return tensor.dtype.as_numpy_dtype().dtype.name


def _prefix_with_scope(name):
    scope = tf.get_variable_scope().name
    return '{}/{}'.format(scope, name) if scope else name


class Variable(TensorMixin, base_wrapper.BaseVariable):
    """Wrap tf.Variable object for storing network parameters"""
    def __init__(self, variable, name=None, trainable=True):
        """Wrap Tensorflow Variable object.

        Parameters
        ----------
        variable : tf.Variable
            Tensorflow Variable object

        name : str or None
            If None, the name is retrieved from variable. Otherwise,
            the given name is prefixed with current scope and used to
            register variable.

        trainable : bool
            Trainable attribute.
        """
        name = _prefix_with_scope(name) if name else variable.op.name
        shape = tuple(variable.get_shape().as_list())
        dtype = _get_dtype_str(variable)
        super(Variable, self).__init__(
            tensor=variable, shape=shape, name=name,
            dtype=dtype, trainable=trainable)


class Tensor(TensorMixin, base_wrapper.BaseTensor):
    """Wrap tf.Tensor object for storing computation result"""
    def __init__(self, tensor, shape=None, name=None):
        """Wrap Tensorflow Tensor object.

        When wrapping Tensor object, as shape and name can be retrieved from
        the object being wrapped, you need not to give them explicitly. You can
        overwrite name attribute for some reasons by giving one. If given, the
        is prefixed with current scope.

        The shape argument is here for compatibility with Theano backend and
        not used in tensorflow backend.

        Parameters
        ----------
        tensor : tf.Tensor
            Tensorflow Tensor object.

        shape
            Not used.

        name : str or None
            If None, the name is retrieved from variable. Otherwise,
            the given name is prefixed with current scope and used to
            register variable.
        """
        name = _prefix_with_scope(name) if name else tensor.name
        shape = tuple(tensor.get_shape().as_list())
        dtype = _get_dtype_str(tensor)
        super(Tensor, self).__init__(
            tensor=tensor, shape=shape, name=name, dtype=dtype)


class Input(TensorMixin, base_wrapper.BaseInput):
    """Represents network input."""
    def __init__(self, shape, name=None, dtype=None):
        """Creates Input object which wraps placeholder

        Parameters
        ----------
        shape : list
            The shape of the resulting object.
        name : str
            The name of the resulting object.
        dtype : NumPy dtype or None
            If None, default dtype is used
        """
        _dtype = dtype or luchador.get_nn_dtype()
        name = _prefix_with_scope(name)
        tensor = tf.placeholder(dtype=_dtype, shape=shape, name=name)
        dtype = _get_dtype_str(tensor)
        super(Input, self).__init__(
            tensor=tensor, shape=shape, name=name, dtype=dtype)


class Operation(base_wrapper.BaseOperation):
    """Wrap tensorflow operations"""
    def __init__(self, op, name=None):
        if luchador.util.is_iteratable(op):
            op = tf.group(*op, name=name)

        name = _prefix_with_scope(name) if name else None
        super(Operation, self).__init__(op=op, name=name)
