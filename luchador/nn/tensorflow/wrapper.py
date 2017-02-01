"""Module for defining input variable/tensor/input wrapper"""
from __future__ import division
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

    def __add__(self, other):
        _other = self._extract_operand(other)
        return Tensor(tensor=self._tensor + _other)

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

    def __truediv__(self, other):
        _other = self._extract_operand(other)
        return Tensor(tensor=self._tensor/_other)

    def __rtruediv__(self, other):
        _other = self._extract_operand(other)
        return Tensor(tensor=_other/self._tensor)

    def __floordiv__(self, other):
        _other = self._extract_operand(other)
        return Tensor(tensor=self._tensor//_other)

    def __rfloordiv__(self, other):
        _other = self._extract_operand(other)
        return Tensor(tensor=_other//self._tensor)

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
            self._tensor, axis=axis, keep_dims=keep_dims, name=name)
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
            self._tensor, axis=axis, keep_dims=keep_dims, name=name)
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
            self._tensor, axis=axis, keep_dims=keep_dims, name=name)
        return Tensor(tensor=_tensor, name=name)

    def clip(self, max_value, min_value, name=None):
        """Clip value elementwise

        Parameters
        ----------
        max_value, min_value : number or Wrapper
            Clip values

        Returns
        -------
        Tensor
            The resulting Tensor
        """
        if isinstance(max_value, base_wrapper.BaseTensor):
            max_value = max_value.unwrap()
        if isinstance(min_value, base_wrapper.BaseTensor):
            min_value = min_value.unwrap()
        _tensor = tf.clip_by_value(
            self._tensor, clip_value_min=min_value, clip_value_max=max_value,
            name=name)
        return Tensor(tensor=_tensor, name=name)

    def one_hot(self, n_classes, dtype=None, name=None):
        """Convert to one-hot encoding.

        Parameters
        ----------
        n_classes : int
            Number of label to encode

        dtype : str
             The dtype of the resulting Tensor. Default to floatX

        name : str
             Name of operation

        Returns
        -------
        Tensor
            Tensor with shape ``(self.shape[0], n_classes)``

        Note
        ----
        The Tensor must be either vector or 2D matrix
        """
        if not self.n_dim == 1:
            raise ValueError('Tensor must be 1D.')

        _dtype = dtype or luchador.get_nn_dtype()
        _tensor = tf.one_hot(
            self._tensor, depth=n_classes, dtype=_dtype, name=name)
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
        _tensor = tf.reshape(self._tensor, shape=new_shape)
        return Tensor(tensor=_tensor, name=name)

    def tile(self, pattern, name=None):
        """Tile tensor.

        Parameters
        ----------
        pattern : tuple
            tile pattern

        Note
        ----
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
        shape = tuple(variable.get_shape().as_list())
        dtype = variable.dtype.as_numpy_dtype().dtype.name
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
        shape = tuple(tensor.get_shape().as_list())
        dtype = tensor.dtype.as_numpy_dtype().dtype.name
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