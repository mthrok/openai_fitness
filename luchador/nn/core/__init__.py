"""Initialize Neural Network module and load backend"""
from __future__ import absolute_import
from .base import (  # noqa
    VariableScope, variable_scope, get_variable_scope, name_scope,
    fetch_node, fetch_layer, fetch_initializer, fetch_optimizer, fetch_cost,
    get_input, get_variable, get_tensor, get_operation, get_grad,
)
from .backend import ops  # noqa
from .impl.wrapper import (  # noqa
    Input, Variable, Tensor, Operation, make_variable,
)
from .impl.random import (  # noqa
    NormalRandom, UniformRandom,
)
from .impl.session import Session, get_session  # noqa
from .impl import (  # noqa
    initializer,
    optimizer,
    layer,
    cost,
)
