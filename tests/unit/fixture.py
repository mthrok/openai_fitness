"""Module for creating test fixture"""
from __future__ import absolute_import

import inspect
import numpy as np

import luchador
import luchador.nn


def create_image(height=210, width=160, channel=3):
    """Create test image"""
    if channel:
        shape = (height, width, channel)
    else:
        shape = (height, width)
    return np.ones(shape, dtype=np.uint8)


def get_all_initializers():
    """Get the all subclasses of BaseInitializer class"""
    class_ = luchador.nn.base.initializer.BaseInitializer
    return {
        name: Class for name, Class
        in inspect.getmembers(luchador.nn.initializer, inspect.isclass)
        if issubclass(Class, class_) and not Class == class_
    }


def get_all_optimizers():
    """Get the all subclasses of BaseOptimizer class"""
    class_ = luchador.nn.base.optimizer.BaseOptimizer
    return {
        name: Class for name, Class
        in inspect.getmembers(luchador.nn.optimizer, inspect.isclass)
        if issubclass(Class, class_) and not Class == class_
    }


def get_all_layers():
    """Get the all subclasses of BaseLayer class"""
    class_ = luchador.nn.base.layer.BaseLayer
    return {
        name: Class for name, Class
        in inspect.getmembers(luchador.nn.layer, inspect.isclass)
        if issubclass(Class, class_) and not Class == class_
    }


def get_all_costs():
    """Get the all subclasses of BaseCost class"""
    class_ = luchador.nn.base.cost.BaseCost
    return {
        name: Class for name, Class
        in inspect.getmembers(luchador.nn.cost, inspect.isclass)
        if issubclass(Class, class_) and not Class == class_
    }
