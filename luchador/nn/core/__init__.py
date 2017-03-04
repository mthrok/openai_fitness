"""Initialize Neural Network module and load backend"""
from __future__ import absolute_import
from .impl.scope import (  # noqa
    VariableScope, variable_scope, get_variable_scope, name_scope,
)
from .impl.wrapper import (  # noqa
    Input, Variable, Tensor, Operation, make_variable,
    get_input, get_variable, get_tensor, get_operation, get_grad
)
from .impl.session import Session  # noqa
from .impl import (  # noqa
    initializer,
    optimizer,
    layer,
    cost,
)
from .backend import ops  # noqa
