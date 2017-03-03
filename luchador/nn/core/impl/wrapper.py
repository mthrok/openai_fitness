"""Implement wrapper classes"""
from __future__ import absolute_import

from ..backend import wrapper

# pylint: disable=invalid-name

Input = wrapper.Input
Variable = wrapper.Variable
Tensor = wrapper.Tensor
Operation = wrapper.Operation
get_variable = wrapper.get_variable
make_variable = wrapper.make_variable

__all__ = [
    'Input', 'Variable', 'Tensor', 'Operation',
    'get_variable', 'make_variable',
]
