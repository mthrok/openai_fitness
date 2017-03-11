"""Define skeleton for network architecture"""
from __future__ import absolute_import

from .base_model import BaseModel, get_model, fetch_model
from .graph import Graph
from .sequential import Sequential
from .container import Container


__all__ = [
    'BaseModel', 'get_model', 'fetch_model',
    'Sequential', 'Graph', 'Container'
]
