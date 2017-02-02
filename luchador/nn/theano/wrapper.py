"""Module for defining input variable/tensor/input wrapper"""
from __future__ import division
from __future__ import absolute_import

import numbers

import theano.tensor as T

import luchador.util
from luchador.nn.base import wrapper as base_wrapper
from luchador.nn.base.wrapper import Operation

__all__ = ['Variable', 'Tensor', 'Input', 'Operation']

_VARIABLES = {}


def _register_variable(name, var):
    if name in _VARIABLES:
        raise ValueError('Variable with name `{}` already exists.')
    _VARIABLES[name] = var


def retrieve_variable(name):
    """Get variable from global list of variables"""
    return _VARIABLES.get(name)


def _is_same_shape(shape1, shape2):
    if not len(shape1) == len(shape2):
        return False

    for dim1, dim2 in zip(shape1, shape2):
        if dim1 is None or dim2 is None:
            continue
        if not dim1 == dim2:
            return False
    return True


def _compute_reduced_shape(axis, shape, keep_dims):
    if not luchador.util.is_iteratable(axis):
        axis = [axis]
    if keep_dims:
        return [
            (1 if i in axis else dim)
            for i, dim in enumerate(shape)]
    return [
        dim for i, dim in enumerate(shape)
        if i not in axis]


def _compute_tile_shape(shape, pattern):
    if len(shape) > len(pattern):
        return _compute_tile_shape(pattern, shape)

    _shape = list(pattern)
    offset = len(pattern) - len(shape)
    for i, val in enumerate(shape):
        if _shape[offset + i] is None:
            continue
        if val is not None:
            _shape[offset + i] *= val
    return _shape


class TensorMixin(object):  # pylint: disable=too-few-public-methods
    """Add elementwise operations to Tensor class"""
    def _extract_operand(self, other):
        if isinstance(other, numbers.Number):
            return other
        if _is_same_shape(self.shape, other.shape):
            return other.unwrap()
        if self.size == 1 or other.size == 1:
            return other.unwrap()
        raise ValueError(
            'Inconsistent shape: {} and {}'.format(self.shape, other.shape)
        )

    def __add__(self, other):
        _other = self._extract_operand(other)
        return Tensor(tensor=self._tensor + _other, shape=self.shape)

    def __sub__(self, other):
        """Scalar subtraction or elementwise subtraction"""
        _other = self._extract_operand(other)
        return Tensor(tensor=self._tensor-_other, shape=self.shape)

    def __rsub__(self, other):
        _other = self._extract_operand(other)
        return Tensor(tensor=_other-self._tensor, shape=self.shape)

    def __mul__(self, other):
        """Scalar multiplication or elementwise multiplication"""
        _other = self._extract_operand(other)
        return Tensor(tensor=self._tensor * _other, shape=self.shape)

    def __truediv__(self, other):
        _other = self._extract_operand(other)
        return Tensor(tensor=self._tensor/_other, shape=self.shape)

    def __rtruediv__(self, other):
        _other = self._extract_operand(other)
        return Tensor(tensor=_other/self._tensor, shape=self.shape)

    def __floordiv__(self, other):
        _other = self._extract_operand(other)
        return Tensor(tensor=self._tensor//_other, shape=self.shape)

    def __rfloordiv__(self, other):
        _other = self._extract_operand(other)
        return Tensor(tensor=_other//self._tensor, shape=self.shape)

    def mean(self, axis=None, keep_dims=False, dtype=None, name=None):
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
        _tensor = self._tensor.mean(axis=axis, keepdims=keep_dims, dtype=dtype)
        _shape = _compute_reduced_shape(axis, self.shape, keep_dims)
        return Tensor(tensor=_tensor, shape=_shape, name=name)

    def sum(self, axis=None, keep_dims=False, dtype=None, name=None):
        """Compute sum across the given axis

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
        _tensor = self._tensor.sum(axis=axis, keepdims=keep_dims, dtype=dtype)
        _shape = _compute_reduced_shape(axis, self.shape, keep_dims)
        return Tensor(tensor=_tensor, shape=_shape, name=name)

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
        _tensor = self._tensor.max(axis=axis, keepdims=keep_dims)
        _shape = _compute_reduced_shape(axis, self.shape, keep_dims)
        return Tensor(tensor=_tensor, shape=_shape, name=name)

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
        _tensor = self._tensor.clip(a_max=max_value, a_min=min_value)
        return Tensor(tensor=_tensor, shape=self.shape, name=name)

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

        _tensor = T.extra_ops.to_one_hot(self._tensor, n_classes, dtype=dtype)
        shape = [self.shape[0], n_classes]
        return Tensor(tensor=_tensor, shape=shape, name=name)

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

        Note
        ----
        This function is for conveniently invoke underlying reshap function.
        Shape-checking and inference is not carried out.
        """
        _tensor = T.reshape(self._tensor, newshape=new_shape)
        return Tensor(tensor=_tensor, shape=new_shape, name=name)

    def tile(self, pattern, name=None):
        """Tile tensor.

        Parameters
        ----------
        pattern : tuple
            tile pattern

        name : str
             Name of operation

        Returns
        -------
        Tensor
            Resulting tensor.

        Note
        ----
        Currently only constant pattern is allowed.
        """
        if not luchador.util.is_iteratable(pattern):
            raise ValueError('`pattern` must be iteratable')

        _shape = _compute_tile_shape(pattern, self.shape)
        _tensor = T.tile(self._tensor, pattern)
        return Tensor(tensor=_tensor, shape=_shape, name=name)


class Variable(TensorMixin, base_wrapper.BaseTensor):
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
        super(Tensor, self).__init__(
            tensor=tensor, shape=shape, name=name, dtype=tensor.dtype)


def _get_tensor(dtype, n_dim, name):
    """Instantiate underlying Variable"""
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


class Input(TensorMixin, base_wrapper.BaseTensor):
    """Represents network input."""
    def __init__(self, shape, name=None, dtype=None):
        """Creates Input object which wraps TensorVariable

        Args:
          shape (list): The shape of the resulting object.
          name (str): The name of the resulting object.
          dtype (NumPy dtype or None): If None, default dtype(floatX) is used
        """
        tensor = _get_tensor(dtype, len(shape), name)
        super(Input, self).__init__(
            tensor=tensor, shape=shape, name=name, dtype=tensor.dtype)
