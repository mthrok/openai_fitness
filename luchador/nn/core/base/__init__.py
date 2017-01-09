"""Implements interface common for NN backends"""
from __future__ import absolute_import

from .layer import BaseLayer, get_layer  # noqa: F401
from .cost import BaseCost, get_cost  # noqa: F401

from .initializer import BaseInitializer, get_initializer  # noqa: F401
from .optimizer import BaseOptimizer, get_optimizer  # noqa: F401
