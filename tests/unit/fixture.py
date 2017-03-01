"""Module for creating test fixture"""
from __future__ import absolute_import

import unittest

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
    class_ = nn.core.base.initializer.BaseInitializer
    return {
        name: Class for name, Class
        in inspect.getmembers(nn.initializer, inspect.isclass)
        if issubclass(Class, class_) and not Class == class_
    }


def get_all_optimizers():
    """Get the all subclasses of BaseOptimizer class"""
    class_ = nn.core.base.optimizer.BaseOptimizer
    return {
        name: Class for name, Class
        in inspect.getmembers(nn.optimizer, inspect.isclass)
        if issubclass(Class, class_) and not Class == class_
    }


def get_all_layers():
    """Get the all subclasses of BaseLayer class"""
    class_ = nn.core.base.layer.BaseLayer
    return {
        name: Class for name, Class
        in inspect.getmembers(nn.layer, inspect.isclass)
        if issubclass(Class, class_) and not Class == class_
    }


def get_all_costs():
    """Get the all subclasses of BaseCost class"""
    class_ = nn.core.base.cost.BaseCost
    return {
        name: Class for name, Class
        in inspect.getmembers(nn.cost, inspect.isclass)
        if issubclass(Class, class_) and not Class == class_
    }


###############################################################################
def create_constant_variable(shape, dtype, value=7, name='constant_varriable'):
    """Create Variable for test"""
    return nn.get_variable(
        name=name, shape=shape, dtype=dtype,
        initializer=nn.initializer.ConstantInitializer(value)
    )


def create_random_variable(
        shape, dtype, min_val=0, max_val=1, name='random_variable'):
    """Create Variable with uniform randoml values for test"""
    return nn.get_variable(
        name=name, shape=shape, dtype=dtype,
        initializer=nn.initializer.UniformInitializer(
            minval=min_val, maxval=max_val)
    )


def create_ones_tensor(shape, dtype, name='ones_tensor'):
    """Create ones Tensor for test in current scope"""
    if luchador.get_nn_backend() == 'theano':
        import theano.tensor as be
    else:
        import tensorflow as be
    tensor = be.ones(shape, dtype=dtype)
    return nn.Tensor(tensor, shape=shape, name=name)


###############################################################################
def gen_scope(test_id, suffix=None):
    """Generate TestCase-specific scope name"""
    scope = test_id.replace('.', '/')
    if suffix:
        scope = '{}/{}'.format(scope, suffix)
    return scope


class TestCase(unittest.TestCase):
    """TestCase with unique scope generation"""
    def get_scope(self, suffix=None):
        """Generate test-case-specific scope name"""
        return gen_scope(self.id().replace('.', '/'), suffix)
