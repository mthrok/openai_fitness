"""Wrap Tensorflow's scoping mechanism for simplar usability"""
from __future__ import absolute_import

import tensorflow as tf

import luchador
from luchador.nn.base import (
    wrapper as base_wrapper,
    initializer as base_initializer,
)
from . import wrapper

__all__ = [
    'VariableScope', 'variable_scope', 'get_variable_scope',
    'name_scope', 'get_variable', 'get_tensor',
]


###############################################################################
# We would want to just rename the function and classes, but Sphynx import the
# docstring from tensorflow and messes up the documentation, so we wrap
# functions and classes, by just passing all the arguments
def name_scope(name, default_name=None, values=None):
    """Wrap Tensorflow name_scope function"""
    return tf.name_scope(name, default_name=default_name, values=values)


def variable_scope(
        name_or_scope, default_name=None, values=None, initializer=None,
        regularizer=None, caching_device=None, partitioner=None,
        custom_getter=None, reuse=None, dtype=None):
    """Wrap Tensorflow variable_scope function"""
    return tf.variable_scope(
        name_or_scope, default_name=default_name, values=values,
        initializer=initializer, regularizer=regularizer,
        caching_device=caching_device, partitioner=partitioner,
        custom_getter=custom_getter, reuse=reuse, dtype=dtype)


def get_variable_scope():
    """Wrap Tensorflow get_variable_scope function"""
    return tf.get_variable_scope()


class VariableScope(tf.VariableScope):  # pylint: disable=R0903
    """Wrap Tensorflow VariableScope class."""
    pass
###############################################################################


def get_tensor(name):
    """Fetch tensor with name in global scope or the current scope

    Parameters
    ----------
    name : str

    Returns
    -------
    Tensor
    """
    try:
        scope = tf.get_variable_scope().name
        return base_wrapper.retrieve_tensor('{}/{}'.format(scope, name))
    except ValueError:
        pass
    return base_wrapper.retrieve_tensor(name)


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

        return wrapper.Variable(variable, trainable=trainable)
