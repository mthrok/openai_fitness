from __future__ import absolute_import

import tensorflow as tf
from tensorflow import (
    name_scope,
    get_variable as _get_variable,
    VariableScope,
    variable_scope,
    get_variable_scope,
)

from luchador import get_nn_dtype
from .wrapper import Variable
from .initializer import TFInitializer

__all__ = ['name_scope', 'get_variable', 'variable_scope',
           'VariableScope', 'get_variable_scope']

###############################################################################
# Mechanism for enabling reusing variable without explicitly giving dtype or
# shape. When creating Variable with get_variable and reuse=False, we store
# mapping from name to (dtype, shape). When retrieving the same Variable with
# get_variable and reuse=True, we use the stored dtype corresponding the given
# name and shape.

_DTYPES = {}


def _register(name, shape, dtype):
    _DTYPES[name] = {'dtype': dtype, 'shape': shape}


def _retrieve(name):
    info = _DTYPES[name]
    return info['dtype'], info['shape']
###############################################################################


def get_variable(name, shape=None, dtype=None,
                 initializer=None, regularizer=None, trainable=True, **kwargs):
    """Create Variable with the given configuration or retrieve existing one

    This function works mostly same as tf.get_variable, except when retrieving
    existing Variable, you only need name and need not to give shape and dtype.

    Mapping from name to shape and dtype is internally cached so that you can
    retrieve variable with only name.

    Args:
      name(str): Name of Variable to create or retrieve

      shape(list): Used to create new Variable.
                   Ignored when retrieving one

      dtype(tf.Dtype compatible): Used to create new Variable.
                                  Ignored when retrieving one

      initializer(TFInitializer or tf.Initializer): Initializer object

      For other arguments, see
      https://www.tensorflow.org/versions/master/api_docs/python/state_ops.html#get_variable
    """
    if isinstance(initializer, TFInitializer):
        initializer = initializer.get()

    if tf.get_variable_scope().reuse:
        dtype, shape = _retrieve(name)
    else:
        dtype = dtype or get_nn_dtype()

    variable = _get_variable(
        name, shape=shape, dtype=dtype, initializer=initializer,
        regularizer=regularizer, trainable=trainable, **kwargs)

    ret = Variable(variable)
    _register(name, ret.shape, ret.dtype)
    return ret
