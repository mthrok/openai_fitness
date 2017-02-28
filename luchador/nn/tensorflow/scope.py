"""Wrap Tensorflow's scoping mechanism for simplar usability"""
from __future__ import absolute_import

import tensorflow as tf

__all__ = [
    'VariableScope', 'variable_scope', 'get_variable_scope', 'name_scope',
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
