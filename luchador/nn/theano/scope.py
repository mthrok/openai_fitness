"""Scoping mechanism similar to Tensorflow"""
from __future__ import absolute_import

import logging
import warnings

import numpy as np
import theano

from luchador.nn.base import wrapper as base_wrapper
from . import wrapper
from .initializer import Normal


__all__ = [
    'VariableScope', 'variable_scope', 'get_variable_scope',
    'name_scope', 'get_variable', 'get_tensor', 'get_input',
]

_LG = logging.getLogger(__name__)


class _NameScope(object):  # pylint: disable=too-few-public-methods
    def __enter__(self):
        pass

    def __exit__(self, type_, value, traceback):
        pass


def name_scope(  # pylint: disable=unused-argument
        name, default_name=None, values=None):
    """Mock Tensorflow name_scope function. Does nothing."""
    return _NameScope()


class VariableScope(object):
    """Mock Tensorflow's VariableScope to provide variable name space"""
    def __init__(self, reuse, name=''):
        self.name = name
        self.reuse = reuse

        self.previous_scopes = []
        self.previous_reuse_flags = []

    @staticmethod
    def reuse_variables():
        """Set reuse flag to True"""
        wrapper.set_flag_(True)

    def _open(self):
        self.previous_scopes.append(wrapper.get_scope_())
        self.previous_reuse_flags.append(wrapper.get_flag_())
        wrapper.set_scope_(self.name)
        wrapper.set_flag_(self.reuse)

    def _close(self):
        wrapper.set_scope_(self.previous_scopes.pop())
        wrapper.set_flag_(self.previous_reuse_flags.pop())

    def __enter__(self):
        # pylint: disable=protected-access
        self._open()
        _LG.debug('Current Scope: %s', wrapper._CURRENT_VARIABLE_SCOPE)
        return self

    def __exit__(self, type_, value, traceback):
        # pylint: disable=protected-access
        self._close()
        _LG.debug('Current Scope: %s', wrapper._CURRENT_VARIABLE_SCOPE)


def variable_scope(name_or_scope, reuse=None):
    """Create new VariableScope object"""
    if isinstance(name_or_scope, VariableScope):
        if reuse:
            return VariableScope(reuse, name_or_scope.name)
        return name_or_scope

    scope = (
        '{}/{}'.format(wrapper.get_scope_(), name_or_scope)
        if wrapper.get_scope_() else name_or_scope
    )
    return VariableScope(reuse, scope)


def get_variable_scope():
    """Return the current variable scope"""
    return VariableScope(wrapper.get_flag_(), wrapper.get_scope_())


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
        scope = wrapper.get_scope_()
        return base_wrapper.retrieve_tensor('{}/{}'.format(scope, name))
    except ValueError:
        pass
    return base_wrapper.retrieve_tensor(name)


def get_input(name):
    """Fetch Input with name in global scope or the current scope

    Parameters
    ----------
    name : str

    Returns
    -------
    Input
    """
    try:
        scope = wrapper.get_scope_()
        return base_wrapper.retrieve_input('{}/{}'.format(scope, name))
    except ValueError:
        pass
    return base_wrapper.retrieve_input(name)


def get_variable(
        name, shape=None, dtype=None, initializer=None,
        regularizer=None, trainable=True, **kwargs):
    """Create/Fetch variable in the current scope

    regularizer is not supported and has no effect.
    """
    if regularizer:
        warnings.warn('`regularizer` is not implemented in Theano backend.')

    # 1. Check the current variable scope
    scope = wrapper.get_scope_()
    name_ = '{}/{}'.format(scope, name) if scope else name

    var = base_wrapper.retrieve_variable(name_)
    if wrapper.get_flag_():  # Search for an existing variable
        if var is None:
            raise ValueError(
                'Variable {} does not exist, disallowed. '
                'Did you mean to set reuse=None in VarScope?'
                .format(name_)
            )
        return var
    else:  # Create new variable
        if var is not None:
            raise ValueError(
                'Variable {} already exists, disallowed. '
                'Did you mean to set reuse=True in VarScope?'
                .format(name_)
            )
        if shape is None:
            raise ValueError(
                'Shape of a new variable ({}) must be fully defined, '
                'but instead was {}.'.format(name_, shape))

        if not initializer:
            initializer = Normal(dtype=dtype)

        # Scalar variable should not have `broadcastable`
        if not shape and 'broadcastable' in kwargs:
            del kwargs['broadcastable']

        return wrapper.Variable(
            theano.shared(
                value=np.array(initializer.sample(shape), dtype=dtype),
                name=name_, allow_downcast=True, **kwargs
            ), trainable=trainable, name=name,
        )
