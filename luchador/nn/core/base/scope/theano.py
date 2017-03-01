"""Scoping mechanism similar to Tensorflow"""
from __future__ import absolute_import

import logging

__all__ = [
    'VariableScope', 'variable_scope', 'get_variable_scope', 'name_scope',
]

_LG = logging.getLogger(__name__)

###############################################################################
_CURRENT_REUSE_FLAG = False
_CURRENT_VARIABLE_SCOPE = ''


def _set_flag(flag):
    """Set reuse flag. Internal user only"""
    # pylint: disable=global-statement
    global _CURRENT_REUSE_FLAG
    _CURRENT_REUSE_FLAG = flag


def _set_scope(scope):
    """Set scope value. Internal user only"""
    # pylint: disable=global-statement
    global _CURRENT_VARIABLE_SCOPE
    _CURRENT_VARIABLE_SCOPE = scope


def _get_flag():
    """Get reuse flag. Internal user only"""
    return _CURRENT_REUSE_FLAG


def _get_scope():
    """Get scope value. Internal user only"""
    return _CURRENT_VARIABLE_SCOPE


def _reset():
    """Reset variable scope and remove cached variables. For Testing"""
    _set_flag(False)
    _set_scope('')
###############################################################################


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
        # pylint: disable=protected-access
        self._open()
        _LG.debug('Current Scope: %s', _CURRENT_VARIABLE_SCOPE)
        return self

    def __exit__(self, type_, value, traceback):
        # pylint: disable=protected-access
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
