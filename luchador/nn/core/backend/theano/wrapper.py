"""Module for defining input variable/tensor/input wrapper"""
from __future__ import division
from __future__ import absolute_import

import numbers
import warnings

import numpy as np
import theano
import theano.tensor as T

from ...base import wrapper as base_wrapper
from ...base import scope as scope_module
from ...base.initializer import get_initializer

__all__ = [
    'Variable', 'Tensor', 'Input', 'Operation', 'make_variable',
]


###############################################################################
def _is_same_shape(shape1, shape2):
    if not len(shape1) == len(shape2):
        return False

    for dim1, dim2 in zip(shape1, shape2):
        if dim1 is None or dim2 is None:
            continue
        if not dim1 == dim2:
            return False
    return True


class TensorMixin(object):  # pylint: disable=too-few-public-methods
    """Add elementwise operations to Tensor class"""
    def _extract_operand(self, other):
        if isinstance(other, numbers.Number):
            return other
        if isinstance(other, base_wrapper.BaseRandomSource):
            return other.sample(shape=self.shape, dtype=self.dtype)
        if _is_same_shape(self.shape, other.shape):
            return other.unwrap()
        if self.size == 1 or other.size == 1:
            return other.unwrap()
        raise ValueError(
            'Inconsistent shape: {} and {}'.format(self.shape, other.shape)
        )

    def __neg__(self, name=None):
        return Tensor(tensor=-self._tensor, shape=self.shape, name=name)

    def __abs__(self, name=None):
        return Tensor(tensor=abs(self.unwrap()), shape=self.shape, name=name)

    def __add__(self, other, name=None):
        _other = self._extract_operand(other)
        return Tensor(tensor=self._tensor+_other, shape=self.shape, name=name)

    def __sub__(self, other, name=None):
        """Scalar subtraction or elementwise subtraction"""
        _other = self._extract_operand(other)
        return Tensor(tensor=self._tensor-_other, shape=self.shape, name=name)

    def __rsub__(self, other, name=None):
        _other = self._extract_operand(other)
        return Tensor(tensor=_other-self._tensor, shape=self.shape, name=name)

    def __mul__(self, other, name=None):
        """Scalar multiplication or elementwise multiplication"""
        _other = self._extract_operand(other)
        return Tensor(tensor=self._tensor*_other, shape=self.shape, name=name)

    def __truediv__(self, other, name=None):
        _other = self._extract_operand(other)
        return Tensor(tensor=self._tensor/_other, shape=self.shape, name=name)

    def __rtruediv__(self, other, name=None):
        _other = self._extract_operand(other)
        return Tensor(tensor=_other/self._tensor, shape=self.shape, name=name)

    def __floordiv__(self, other, name=None):
        _other = self._extract_operand(other)
        return Tensor(tensor=self._tensor//_other, shape=self.shape, name=name)

    def __rfloordiv__(self, other, name=None):
        _other = self._extract_operand(other)
        return Tensor(tensor=_other//self._tensor, shape=self.shape, name=name)


def _get_scope():
    return scope_module.get_variable_scope()


def _prefix_with_scope(name):
    scope = _get_scope().name
    return '{}/{}'.format(scope, name) if scope else name


class Variable(TensorMixin, base_wrapper.BaseVariable):
    """Wrap SharedVariable object for storing network parameters"""
    def __init__(self, variable, name=None, trainable=True):
        """Wrap SharedVariable object.

        Args:
          variable (SharedVariable): theano.tensor.SharedVariable object
          name (str or None): When given, the name of the resulting wrapper is
            overwritten with this name, otherwise, name is constructed in the
            manner as Tensorflow.
        """
        name = _prefix_with_scope(name or variable.name)
        val = variable.get_value()
        super(Variable, self).__init__(
            tensor=variable, shape=val.shape, name=name,
            dtype=val.dtype, trainable=trainable)


class Tensor(TensorMixin, base_wrapper.BaseTensor):
    """Wrap TensorVariable object for storing computation result"""
    def __init__(self, tensor, shape=None, name=None):
        """Wrap TensorVariable object.

        Args:
          tensor (TensorVariable): theano.tensor.TensorVariable object
          shape (list): Shape of the tensor being wrapped.
          name (str or None): Name of the resulting wrapper for convenience.
        """
        if -1 in shape:
            shape = [None if val < 0 else val for val in shape]
        name = _prefix_with_scope(name) if name else None
        super(Tensor, self).__init__(
            tensor=tensor, shape=shape, name=name, dtype=tensor.dtype)


def _create_placeholder(dtype, n_dim, name):
    if n_dim == 0:
        tensor = T.scalar(name=name, dtype=dtype)
    elif n_dim == 1:
        tensor = T.vector(name=name, dtype=dtype)
    elif n_dim == 2:
        tensor = T.matrix(name=name, dtype=dtype)
    elif n_dim == 3:
        tensor = T.tensor3(name=name, dtype=dtype)
    elif n_dim == 4:
        tensor = T.tensor4(name=name, dtype=dtype)
    else:
        raise ValueError('shape length must be smaller than 5')
    return tensor


class Input(TensorMixin, base_wrapper.BaseInput):
    """Represents network input."""
    def __init__(self, shape, name=None, dtype=None):
        """Creates Input object which wraps TensorVariable

        Args:
          shape (list): The shape of the resulting object.
          name (str): The name of the resulting object.
          dtype (NumPy dtype or None): If None, default dtype(floatX) is used
        """
        name = _prefix_with_scope(name) if name else None
        tensor = _create_placeholder(dtype, len(shape), name)
        super(Input, self).__init__(
            tensor=tensor, shape=shape, name=name, dtype=tensor.dtype)


class Operation(base_wrapper.BaseOperation):
    """Represents operation"""
    def __init__(self, op, name=None):
        name = _prefix_with_scope(name) if name else None
        super(Operation, self).__init__(op=op, name=name)


def make_variable(
        name, shape, dtype=None,
        initializer=None, regularizer=None, trainable=True, **kwargs):
    """Create Variable with the given configuration

    Parameters
    ----------
    name : str
        Name of Variable to create or retrieve

    shape : list
        Used to create new Variable. Ignored when retrieving one

    dtype : str
        Used to create new Variable. Ignored when retrieving one

    initializer : luchador.nn.Initializer or tf.Initializer
        Initializer object

    kwargs
        Other arguments passed to ``theano.shared`` function.
    """
    scope = _get_scope().name
    name_ = '{}/{}'.format(scope, name) if scope else name

    if not initializer:
        initializer = get_initializer('NormalInitializer')(dtype=dtype)

    if regularizer:
        warnings.warn('`regularizer` is not implemented in Theano backend.')

    # Scalar variable should not have `broadcastable`
    if not shape and 'broadcastable' in kwargs:
        del kwargs['broadcastable']

    return Variable(
        theano.shared(
            value=np.array(initializer.sample(shape), dtype=dtype),
            name=name_, allow_downcast=True, **kwargs
        ), name=name, trainable=trainable,
    )
