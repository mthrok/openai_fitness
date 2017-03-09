"""Defines and exports scoping mechanism, base classes and getter."""
from __future__ import absolute_import

from .scope import (
    VariableScope, variable_scope, get_variable_scope, name_scope
)
from .node import Node, get_node
from .wrapper import (
    BaseWrapper,
    BaseInput, get_input,
    BaseVariable, get_variable,
    BaseTensor, get_tensor, get_grad,
    BaseOperation, get_operation,
)
from .cost import BaseCost, get_cost
from .layer import BaseLayer, get_layer
from .optimizer import BaseOptimizer, get_optimizer
from .initializer import BaseInitializer, get_initializer


__all__ = [
    'VariableScope', 'variable_scope', 'get_variable_scope', 'name_scope',
    'Node', 'get_node',
    'BaseWrapper',
    'BaseInput', 'get_input',
    'BaseVariable', 'get_variable',
    'BaseTensor', 'get_tensor', 'get_grad',
    'BaseOperation', 'get_operation',
    'BaseLayer', 'get_layer',
    'BaseInitializer', 'get_initializer',
    'BaseOptimizer', 'get_optimizer',
    'BaseCost', 'get_cost',
]
