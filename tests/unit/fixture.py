"""Module for creating test fixture"""
from __future__ import absolute_import

import inspect
import numpy as np

import luchador
from luchador import nn


def create_image(height=210, width=160, channel=3):
    """Create test image"""
    if channel:
        shape = (height, width, channel)
    else:
        shape = (height, width)
    return np.ones(shape, dtype=np.uint8)


def get_all_initializers():
    """Get the all subclasses of BaseInitializer class"""
    class_ = nn.base.initializer.BaseInitializer
    return {
        name: Class for name, Class
        in inspect.getmembers(nn.initializer, inspect.isclass)
        if issubclass(Class, class_) and not Class == class_
    }


def get_all_optimizers():
    """Get the all subclasses of BaseOptimizer class"""
    class_ = nn.base.optimizer.BaseOptimizer
    return {
        name: Class for name, Class
        in inspect.getmembers(nn.optimizer, inspect.isclass)
        if issubclass(Class, class_) and not Class == class_
    }


def get_all_layers():
    """Get the all subclasses of BaseLayer class"""
    class_ = nn.base.layer.BaseLayer
    return {
        name: Class for name, Class
        in inspect.getmembers(nn.layer, inspect.isclass)
        if issubclass(Class, class_) and not Class == class_
    }


def get_all_costs():
    """Get the all subclasses of BaseCost class"""
    class_ = nn.base.cost.BaseCost
    return {
        name: Class for name, Class
        in inspect.getmembers(nn.cost, inspect.isclass)
        if issubclass(Class, class_) and not Class == class_
    }


###############################################################################
def create_variable(shape, value=7, dtype='int32', name='var'):
    """Create Variable for test"""
    return nn.get_variable(
        name=name, shape=shape, dtype=dtype,
        initializer=nn.initializer.Constant(value)
    )


def create_tensor(shape, dtype='int32', name='tensor'):
    """Create Tensor for test"""
    if luchador.get_nn_backend() == 'theano':
        import theano.tensor as be
    else:
        import tensorflow as be
    tensor = be.ones(shape, dtype=dtype)
    return nn.Tensor(tensor, shape=shape, name=name)
