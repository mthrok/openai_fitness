from __future__ import absolute_import

import os
import warnings

import numpy as np

_NN_DTYPE = os.environ.get('LUCHADOR_NN_DTYPE', 'float32')
_NN_BACKEND = os.environ.get('LUCHADOR_NN_BACKEND', 'theano')
_NN_CONV_FORMAT = os.environ.get('LUCHADOR_NN_CONV_FORMAT', 'NCHW')

__all__ = [
    'get_nn_backend', 'get_nn_conv_format', 'get_nn_dtype',
    'set_nn_backend', 'set_nn_conv_format', 'set_nn_dtype',
]


def get_nn_backend():
    return _NN_BACKEND


def get_nn_conv_format():
    return _NN_CONV_FORMAT


def get_nn_dtype():
    return _NN_DTYPE


def set_nn_backend(backend):
    global _NN_BACKEND
    _NN_BACKEND = backend


def set_nn_conv_format(fmt):
    if fmt not in ('NCHW', 'NHWC'):
        raise ValueError('Convolution format must b either "NCHW" or "NHWC"')

    if _NN_BACKEND == 'theano':
        warnings.warn('Convolution format only affects "tensorflow" backend')

    global _NN_CONV_FORMAT
    _NN_CONV_FORMAT = fmt


def set_nn_dtype(dtype):
    global _NN_DTYPE

    if _NN_BACKEND == 'theano':
        msg = (
            'In theano backend, theano.config.floatX is used, '
            'thus setting dtype has no effect.'
        )
        warnings.warn(msg)

    dtype_ = np.dtype(dtype)
    _NN_DTYPE = dtype_.name
