"""Defines interface for NN components such as layer, optimizer"""
from __future__ import absolute_import

from .cost import get_cost  # noqa: F401
from .layer import get_layer  # noqa: F401
from .optimizer import get_optimizer  # noqa: F401
from .initializer import get_initializer  # noqa: F401
