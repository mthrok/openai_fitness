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
from .wrapper import (
    Variable,
    retrieve_variable,
)
from .initializer import TFInitializer

__all__ = ['name_scope', 'get_variable', 'variable_scope',
           'VariableScope', 'get_variable_scope']


def get_variable(name, shape=None, dtype=None,
                 initializer=None, regularizer=None, trainable=True, **kwargs):
    """Create Variable with the given configuration or retrieve existing one

    This function works mostly same as tf.get_variable, except when retrieving
    existing Variable, you only need name and need not to give shape and dtype.

    Mapping from name to VariableWrapper is internally cached so that you can
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
        initializer = initializer.unwrap()

    scope = tf.get_variable_scope()
    if scope.reuse:
        name = '{}/{}'.format(scope.name, name) if scope.name else name
        var = retrieve_variable(name)
        if var is None:
            raise ValueError(
                'Variable {} does not exist, disallowed. '
                'Did you mean to set reuse=None in VarScope?'
                .format(name)
            )
        return var
    else:
        dtype = dtype or get_nn_dtype()

        variable = _get_variable(
            name, shape=shape, dtype=dtype, initializer=initializer,
            regularizer=regularizer, trainable=trainable, **kwargs)

        return Variable(variable, trainable=trainable)
