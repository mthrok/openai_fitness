"""Module for configuring luchador library"""
from __future__ import absolute_import

import os
import warnings

import numpy as np

# pylint: disable=global-statement

_NN_DTYPE = os.environ.get('LUCHADOR_NN_DTYPE', 'float32')
_NN_BACKEND = os.environ.get('LUCHADOR_NN_BACKEND', 'theano')
_NN_CONV_FORMAT = os.environ.get('LUCHADOR_NN_CONV_FORMAT', 'NCHW')

__all__ = [
    'get_nn_backend', 'get_nn_conv_format', 'get_nn_dtype',
    'set_nn_backend', 'set_nn_conv_format', 'set_nn_dtype',
]


def get_nn_backend():
    """Get NN module backend

    Returns
    -------
    str
        Either ``theano`` or ``tensorflow``
    """
    return _NN_BACKEND


def get_nn_conv_format():
    """Get convolution data format of Tensorflow backend

    Returns
    -------
    str
        Either ``NCHW`` or ``NHCW``.
        Refer to ``tf.nn.conv2d`` of the official Tensorflow documentation.
    """
    return _NN_CONV_FORMAT


def get_nn_dtype():
    """Get default dtype of Tensorflow backend

    Returns
    -------
    str
        The name of default ``tf.DType``
    """
    return _NN_DTYPE


def set_nn_backend(backend):
    """Set NN module backend

    .. note:: Backend must be set before ``luchador.nn module`` is imported.

    Parameters
    ----------
    backend : str
        Either ``theano`` or ``tensorflow``
    """
    if backend not in ('theano', 'tensorflow'):
        raise ValueError('NN Backend must be either "theano" or "tensorflow"')

    global _NN_BACKEND
    _NN_BACKEND = backend


def set_nn_conv_format(format_):
    """Set convolution data format of Tensorflow

    .. note::
        Convolution format has affect only in Tensorflow backend.

    Parameters
    ----------
    format_ : str
        Either ``NHCW`` or ``NCHW``. See``tf.nn.conv2d`` for the detail.
    """
    if format_ not in ('NCHW', 'NHWC'):
        raise ValueError('Convolution format must b either "NCHW" or "NHWC"')

    if _NN_BACKEND == 'theano':
        warnings.warn('Convolution format only affects "tensorflow" backend')

    global _NN_CONV_FORMAT
    _NN_CONV_FORMAT = format_


def set_nn_dtype(dtype):
    """Set default dtype for creating variables in Tensorflow backend

    .. note::
        This dtype has affect only in Tensorflow backend.
        To set default dtype of Theano backend, use ``floatX``.

    Parameters
    ----------
    dtype : str
        String expression of ``tf.DType``.
    """
    global _NN_DTYPE
    if _NN_BACKEND == 'theano':
        msg = (
            'In theano backend, theano.config.floatX is used, '
            'thus setting dtype has no effect.'
        )
        warnings.warn(msg)

    dtype_ = np.dtype(dtype)
    _NN_DTYPE = dtype_.name
