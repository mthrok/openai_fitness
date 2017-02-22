"""Module for defining input variable/tensor/input wrapper"""
from __future__ import division
from __future__ import absolute_import

import numbers

import theano.tensor as T

import luchador.util
from ..base import wrapper as base_wrapper

__all__ = ['Variable', 'Tensor', 'Input', 'Operation']


###############################################################################
_CURRENT_REUSE_FLAG = False
_CURRENT_VARIABLE_SCOPE = ''


def set_flag_(flag):
    """Set reuse flag. Internal user only"""
    # pylint: disable=global-statement
    global _CURRENT_REUSE_FLAG
    _CURRENT_REUSE_FLAG = flag


def set_scope_(scope):
    """Set scope value. Internal user only"""
    # pylint: disable=global-statement
    global _CURRENT_VARIABLE_SCOPE
    _CURRENT_VARIABLE_SCOPE = scope


def get_flag_():
    """Get reuse flag. Internal user only"""
    return _CURRENT_REUSE_FLAG


def get_scope_():
    """Get scope value. Internal user only"""
    return _CURRENT_VARIABLE_SCOPE


def _reset():
    """Reset variable scope and remove cached variables. For Testing"""
    set_flag_(False)
    set_scope_('')


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
        _tensor = self.unwrap().mean(
            axis=axis, keepdims=keep_dims, dtype=dtype)
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
        _tensor = self.unwrap().sum(axis=axis, keepdims=keep_dims, dtype=dtype)
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
        _tensor = self.unwrap().max(axis=axis, keepdims=keep_dims)
        _shape = _compute_reduced_shape(axis, self.shape, keep_dims)
        return Tensor(tensor=_tensor, shape=_shape, name=name)

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

        Notes
        -----
        This function is for conveniently invoke underlying reshap function.
        Shape-checking and inference is not carried out.
        """
        _tensor = T.reshape(self.unwrap(), newshape=new_shape)
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

        Notes
        -----
        Currently only constant pattern is allowed.
        """
        if not luchador.util.is_iteratable(pattern):
            raise ValueError('`pattern` must be iteratable')

        _shape = _compute_tile_shape(pattern, self.shape)
        _tensor = T.tile(self.unwrap(), pattern)
        return Tensor(tensor=_tensor, shape=_shape, name=name)


def _prefix_with_scope(name):
    scope_ = get_scope_()
    return '{}/{}'.format(scope_, name) if scope_ else name


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
