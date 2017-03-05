"""Implement mechanism to store/fetch Input/Variable/Tensor/Operation"""
from __future__ import absolute_import

import logging
from collections import OrderedDict

__all__ = ['register', 'retrieve']
_LG = logging.getLogger(__name__)
# pylint: disable=redefined-builtin


###############################################################################
# Mechanism for fetching Variable, Tensor, Input and Operation only by name
# This is similar to Tensorflow's get_variable function with reuse=Trie, but
# 1. Tensorflow's get_variable require dtype and shape for retrieving
# exisiting varaible, which is inconvenient.
# 2. It is extendef to Input, Tensor and Operation objects
_VARIABLES = OrderedDict()
_TENSORS = OrderedDict()
_INPUTS = OrderedDict()
_OPERATIONS = OrderedDict()


def _register_variable(name, var):
    if name in _VARIABLES:
        raise ValueError('Variable `{}` already exists.'.format(name))
    _VARIABLES[name] = var


def _register_tensor(name, tensor):
    if name in _TENSORS:
        _LG.warning('Tensor `%s` already exists.', name)
    _TENSORS[name] = tensor


def _register_input(name, input_):
    if name in _INPUTS:
        _LG.warning('Input `%s` already exists.', name)
    _INPUTS[name] = input_


def _register_operation(name, operation):
    if name in _OPERATIONS:
        _LG.warning('Operation `%s` already exists.', name)
    _OPERATIONS[name] = operation


def _retrieve_variable(name):
    if name not in _VARIABLES:
        raise ValueError('Variable `{}` does not exist.'.format(name))
    return _VARIABLES.get(name)


def _retrieve_tensor(name):
    if name not in _TENSORS:
        raise ValueError('Tensor `{}` does not exist.'.format(name))
    return _TENSORS.get(name)


def _retrieve_input(name):
    if name not in _INPUTS:
        raise ValueError('Input `{}` does not exist.'.format(name))
    return _INPUTS[name]


def _retrieve_operation(name):
    if name not in _OPERATIONS:
        raise ValueError('Operation `{}` does not exist.'.format(name))
    return _OPERATIONS[name]


def register(type, name, obj):
    """Register object"""
    if type == 'input':
        _register_input(name, obj)
    elif type == 'variable':
        _register_variable(name, obj)
    elif type == 'tensor':
        _register_tensor(name, obj)
    elif type == 'operation':
        _register_operation(name, obj)
    else:
        raise ValueError('Unknown type to register {}'.format(type))


def retrieve(type, name):
    """Retrieve object"""
    if type == 'input':
        return _retrieve_input(name)
    elif type == 'variable':
        return _retrieve_variable(name)
    elif type == 'tensor':
        return _retrieve_tensor(name)
    elif type == 'operation':
        return _retrieve_operation(name)
    else:
        raise ValueError('Unknown type to retrieve {}'.format(type))
