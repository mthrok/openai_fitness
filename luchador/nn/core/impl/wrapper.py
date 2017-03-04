"""Implement wrapper classes"""
from __future__ import absolute_import

from ..base.wrapper import (
    get_input, get_variable, get_tensor, get_operation, get_grad
)
from ..backend import wrapper

# pylint: disable=invalid-name

Input = wrapper.Input
Variable = wrapper.Variable
Tensor = wrapper.Tensor
Operation = wrapper.Operation
make_variable = wrapper.make_variable

__all__ = [
    'Input', 'Variable', 'Tensor', 'Operation', 'make_variable',
    'get_input', 'get_variable', 'get_tensor', 'get_operation', 'get_grad',
]
