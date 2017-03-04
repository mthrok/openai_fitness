"""Module for defining input variable/tensor/input wrapper"""
from __future__ import division
from __future__ import absolute_import

import numbers

import tensorflow as tf

import luchador
import luchador.util
from ...base import wrapper as base_wrapper
from ...base import scope as scope_module
from ...base.initializer import BaseInitializer, get_initializer

__all__ = [
    'Variable', 'Tensor', 'Input', 'Operation', 'make_variable',
]


class TensorMixin(object):  # pylint: disable=too-few-public-methods
    """Add elementwise operations to Tensor class"""
    def _extract_operand(self, other):
        """Extract operand for elementwise operation"""
        if isinstance(other, numbers.Number):
            return other
        return other.unwrap()

    def __neg__(self, name=None):
        return Tensor(tensor=-self._tensor, name=name)

    def __abs__(self, name=None):
        return Tensor(tensor=tf.abs(self._tensor), name=name)

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


def _get_dtype_str(tensor):
    return tensor.dtype.as_numpy_dtype().dtype.name


def _get_scope():
    return scope_module.get_variable_scope()


def _is_reuse():
    return _get_scope().reuse


def _prefix_with_scope(name):
    scope = _get_scope().name
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
        tensor = tf.placeholder(dtype=_dtype, shape=shape, name=name)
        name = _prefix_with_scope(name)
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


def make_variable(
        name, shape, dtype=None,
        initializer=None, regularizer=None, trainable=True, **kwargs):
    """Create Variable with the given configuration

    Parameters
    ----------
    name : str
        Name of Variable to create.

    shape : list
        Used to create new Variable.

    dtype : str
        Used to create new Variable.

    initializer : luchador.nn.Initializer
        Initializer object

    kwargs
        Other arguments passed to ``tf.get_variable``
        See
        https://www.tensorflow.org/versions/master/api_docs/python/state_ops.html#get_variable
    """
    dtype = dtype or luchador.get_nn_dtype()

    if not initializer:
        initializer = get_initializer('NormalInitializer')(dtype=dtype)

    if isinstance(initializer, BaseInitializer):
        initializer = initializer.unwrap()

    return Variable(
        tf.get_variable(
            name, shape=shape, dtype=dtype, initializer=initializer,
            regularizer=regularizer, trainable=trainable, **kwargs
        ), name=name, trainable=trainable
    )
