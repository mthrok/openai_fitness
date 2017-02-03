"""Defines interface for NN components such as layer, optimizer"""
from __future__ import absolute_import
# pylint: disable=wildcard-import
from .cost import get_cost, BaseCost  # noqa: F401
from .layer import get_layer, BaseLayer  # noqa: F401
from .wrapper import *  # noqa: F401, F403
from .optimizer import get_optimizer, BaseOptimizer  # noqa: F401
from .initializer import get_initializer, BaseInitializer  # noqa: F401
