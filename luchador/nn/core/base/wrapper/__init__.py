"""Define base wrapper classes"""
from __future__ import absolute_import

from .random import BaseRandomSource
from .wrapper import BaseWrapper
from .input import BaseInput, get_input
from .variable import BaseVariable, get_variable
from .tensor import BaseTensor, get_tensor, get_grad
from .operation import BaseOperation, get_operation

__all__ = [
    'BaseRandomSource', 'BaseWrapper', 'BaseTensor', 'BaseVariable',
    'BaseInput', 'BaseOperation',
    'get_input', 'get_variable', 'get_tensor', 'get_operation', 'get_grad',
]
