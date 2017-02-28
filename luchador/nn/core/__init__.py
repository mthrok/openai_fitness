"""Initialize Neural Network module and load backend"""
from __future__ import absolute_import
from .base import (  # noqa
    VariableScope, variable_scope, get_variable_scope, name_scope,
    get_node, get_input, get_variable, get_tensor, get_operation, get_grad,
    get_layer, get_initializer, get_optimizer, get_cost,
)
from .backend import ops  # noqa
from .impl.wrapper import (  # noqa
    Input, Variable, Tensor, Operation, make_variable,
)
from .impl.random import (  # noqa
    NormalRandom, UniformRandom,
)
from .impl.session import Session  # noqa
from .impl import (  # noqa
    initializer,
    optimizer,
    layer,
    cost,
)
