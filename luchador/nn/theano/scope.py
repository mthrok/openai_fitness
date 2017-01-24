"""Scoping mechanism similar to Tensorflow"""
from __future__ import absolute_import

import logging
import warnings

import numpy as np
import theano
from theano import config as theano_config

from . import wrapper
from .initializer import Normal

__all__ = [
    'VariableScope', 'variable_scope', 'get_variable_scope',
    'name_scope', 'get_variable',
]

_LG = logging.getLogger(__name__)

_CURRENT_REUSE_FLAG = False
_CURRENT_VARIABLE_SCOPE = ''


def _set_flag(flag):
    # pylint: disable=global-statement
    global _CURRENT_REUSE_FLAG
    _CURRENT_REUSE_FLAG = flag


def _set_scope(scope):
    # pylint: disable=global-statement
    global _CURRENT_VARIABLE_SCOPE
    _CURRENT_VARIABLE_SCOPE = scope


def _get_flag():
    return _CURRENT_REUSE_FLAG


def _get_scope():
    return _CURRENT_VARIABLE_SCOPE


def _reset():
    """Reset variable scope and remove cached variables. For Testing"""
    # pylint: disable=protected-access
    _set_flag(False)
    _set_scope('')
    wrapper._VARIABLES = {}


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
        _set_flag(True)

    def _open(self):
        self.previous_scopes.append(_get_scope())
        self.previous_reuse_flags.append(_get_flag())
        _set_scope(self.name)
        _set_flag(self.reuse)

    def _close(self):
        _set_scope(self.previous_scopes.pop())
        _set_flag(self.previous_reuse_flags.pop())

    def __enter__(self):
        self._open()
        _LG.debug('Current Scope: %s', _CURRENT_VARIABLE_SCOPE)
        return self

    def __exit__(self, type_, value, traceback):
        self._close()
        _LG.debug('Current Scope: %s', _CURRENT_VARIABLE_SCOPE)


def variable_scope(name_or_scope, reuse=None):
    """Create new VariableScope object"""
    if isinstance(name_or_scope, VariableScope):
        if reuse:
            return VariableScope(reuse, name_or_scope.name)
        return name_or_scope

    scope = (
        '{}/{}'.format(_get_scope(), name_or_scope)
        if _get_scope() else name_or_scope
    )
    return VariableScope(reuse, scope)


def get_variable_scope():
    """Return the current variable scope"""
    return VariableScope(_get_flag(), _get_scope())


def get_variable(
        name, shape=None, dtype=None, initializer=None,
        regularizer=None, trainable=True, **kwargs):
    """Create/Fetch variable in the current scope

    regularizer is not supported and has no effect.
    """
    if regularizer:
        warnings.warn('`regularizer` is not implemented in Theano backend.')

    # 1. Check the current variable scope
    scope = _get_scope()
    name = '{}/{}'.format(scope, name) if scope else name

    var = wrapper.retrieve_variable(name)
    if _get_flag():  # Search for an existing variable
        if var is None:
            raise ValueError(
                'Variable {} does not exist, disallowed. '
                'Did you mean to set reuse=None in VarScope?'
                .format(name)
            )
        return var
    else:  # Create new variable
        if var is not None:
            raise ValueError(
                'Variable {} already exists, disallowed. '
                'Did you mean to set reuse=True in VarScope?'
                .format(name)
            )
        if shape is None:
            raise ValueError(
                'Shape of a new variable ({}) must be fully defined, '
                'but instead was {}.'.format(name, shape))

        dtype = dtype or theano_config.floatX

        if not initializer:
            initializer = Normal(dtype=dtype)

        # Scalar variable should not have `broadcastable`
        if not shape and 'broadcastable' in kwargs:
            del kwargs['broadcastable']

        return wrapper.Variable(
            theano.shared(
                value=np.array(initializer.sample(shape), dtype=dtype),
                name=name, allow_downcast=True, **kwargs
            ), trainable=trainable,
        )
