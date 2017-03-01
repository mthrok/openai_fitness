"""Module for defining input variable/tensor/input wrapper"""
from __future__ import division
from __future__ import absolute_import

import numbers

import tensorflow as tf

import luchador
import luchador.util
from ..base import (
    wrapper as base_wrapper,
    initializer as base_initializer,
)

__all__ = ['Variable', 'Tensor', 'Input', 'Operation']


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

    def mean(self, axis=None, keep_dims=False, name=None):
        """Compute mean across the given axis

        Parameters
        ----------
        axis : int, list or None
            The dimensions to compute mean. If None (the default),
            reduces all dimensions.
        keep_dims: bool
            If true, retains reduced dimensions with length 1.
        name: str
            A name for the operation.

        Returns
        -------
        Tensor
            The resulting Tensor
        """
        _tensor = tf.reduce_mean(
            self.unwrap(), axis=axis, keep_dims=keep_dims, name=name)
        return Tensor(tensor=_tensor, name=name)

    def sum(self, axis=None, keep_dims=False, name=None):
        """Compute sum across the given axis

        Parameters
        ----------
        axis : int, list or None
            The dimensions to compute sum. If None (the default),
            reduces all dimensions.
        keep_dims: bool
            If true, retains reduced dimensions with length 1.
        name: str
            A name for the operation.

        Returns
        -------
        Tensor
            The resulting Tensor
        """
        _tensor = tf.reduce_sum(
            self.unwrap(), axis=axis, keep_dims=keep_dims, name=name)
        return Tensor(tensor=_tensor, name=name)

    def max(self, axis=None, keep_dims=False, name=None):
        """Compute max across the given axis

        Parameters
        ----------
        axis : int, list or None
            The dimensions to compute max. If None (the default),
            reduces all dimensions.
        keep_dims: bool
            If true, retains reduced dimensions with length 1.
        name: str
            A name for the operation.

        Returns
        -------
        Tensor
            The resulting Tensor
        """
        _tensor = tf.reduce_max(
            self.unwrap(), axis=axis, keep_dims=keep_dims, name=name)
        return Tensor(tensor=_tensor, name=name)

    def reshape(self, new_shape, name=None):
        """Reshape tensor.

        Parameters
        ----------
        new_shape : tuple
            new shape

        name : str
            Name of operation

        Returns
        -------
        Tensor
            Tensor with new shape
        """
        _tensor = tf.reshape(self.unwrap(), shape=new_shape)
        return Tensor(tensor=_tensor, name=name)

    def tile(self, pattern, name=None):
        """Tile tensor.

        Parameters
        ----------
        pattern : tuple
            tile pattern

        Notes
        -----
        Currently only constant pattern is allowed.
        """
        if not luchador.util.is_iteratable(pattern):
            raise ValueError('`pattern` must be iteratable')
        pattern = tuple(pattern)

        if len(pattern) > self.n_dim:
            prepend = (1, ) * (len(pattern) - self.n_dim)
            tensor = self.reshape(prepend + self.shape).unwrap()
        else:
            prepend = (1, ) * (self.n_dim - len(pattern))
            pattern = prepend + pattern
            tensor = self.unwrap()
        return Tensor(tf.tile(tensor, pattern, name), name=name)


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


def get_variable(
        name, shape=None, dtype=None,
        initializer=None, regularizer=None, trainable=True, **kwargs):
    """Create Variable with the given configuration or retrieve existing one

    This function works mostly same as tf.get_variable, except when retrieving
    existing Variable, you only need name and need not to give shape and dtype.

    Mapping from name to VariableWrapper is internally cached so that you can
    retrieve variable with only name.

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
        Other arguments passed to ``tf.get_variable``
        See
        https://www.tensorflow.org/versions/master/api_docs/python/state_ops.html#get_variable
    """
    if isinstance(initializer, base_initializer.BaseInitializer):
        initializer = initializer.unwrap()

    scope = tf.get_variable_scope()
    if scope.reuse:
        name = '{}/{}'.format(scope.name, name) if scope.name else name
        var = base_wrapper.retrieve_variable(name)
        if var is None:
            raise ValueError(
                'Variable {} does not exist, disallowed. '
                'Did you mean to set reuse=None in VarScope?'
                .format(name)
            )
        return var
    else:
        dtype = dtype or luchador.get_nn_dtype()

        variable = tf.get_variable(
            name, shape=shape, dtype=dtype, initializer=initializer,
            regularizer=regularizer, trainable=trainable, **kwargs)

        return Variable(variable, trainable=trainable)
