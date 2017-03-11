"""Defines and exports scoping mechanism, base classes and getter."""
from __future__ import absolute_import

from .scope import (
    VariableScope, variable_scope, get_variable_scope, name_scope
)
from .node import Node, fetch_node
from .wrapper import (
    BaseWrapper,
    BaseInput, get_input,
    BaseVariable, get_variable,
    BaseTensor, get_tensor, get_grad,
    BaseOperation, get_operation,
)
from .cost import BaseCost, fetch_cost
from .layer import BaseLayer, fetch_layer
from .optimizer import BaseOptimizer, fetch_optimizer
from .initializer import BaseInitializer, fetch_initializer


__all__ = [
    'VariableScope', 'variable_scope', 'get_variable_scope', 'name_scope',
    'Node', 'fetch_node',
    'BaseWrapper',
    'BaseInput', 'get_input',
    'BaseVariable', 'get_variable',
    'BaseTensor', 'get_tensor', 'get_grad',
    'BaseOperation', 'get_operation',
    'BaseLayer', 'fetch_layer',
    'BaseInitializer', 'fetch_initializer',
    'BaseOptimizer', 'fetch_optimizer',
    'BaseCost', 'fetch_cost',
]
