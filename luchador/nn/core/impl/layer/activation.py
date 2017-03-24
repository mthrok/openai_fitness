"""Define common interface for Layer classes"""
from __future__ import division
from __future__ import absolute_import

from ...base.layer import BaseLayer
from ...backend import layer

__all__ = ['ReLU', 'LeakyReLU', 'Softplus', 'Sigmoid', 'Tanh', 'Softmax']


class ReLU(layer.ReLU, BaseLayer):
    """Apply rectified linear activation elementwise

    Parameters
    ----------
    name : str
        Used as base scope when building parameters and output
    """
    def __init__(self, name='ReLU'):
        super(ReLU, self).__init__(name=name)


class LeakyReLU(layer.LeakyReLU, BaseLayer):
    """Apply rectified linear activation elementwise

    Parameters
    ----------
    alpha : float
        Initial value of slope for negative region.

    train : bool
        When True, ``alpha`` variable becomes trainable variable.

    name : str
        Used as base scope when building parameters and output
    """
    def __init__(self, alpha, train=False, name='LeakyReLU'):
        super(LeakyReLU, self).__init__(name=name, train=train, alpha=alpha)


class Sigmoid(layer.Sigmoid, BaseLayer):
    """Apply sigmoid activation elementwise

    Parameters
    ----------
    name : str
        Used as base scope when building parameters and output
    """
    def __init__(self, name='Sigmoid'):
        super(Sigmoid, self).__init__(name=name)


class Tanh(layer.Tanh, BaseLayer):
    """Apply tanh activation elementwise

    Parameters
    ----------
    name : str
        Used as base scope when building parameters and output
    """
    def __init__(self, name='Tanh'):
        super(Tanh, self).__init__(name=name)


class Softmax(layer.Softmax, BaseLayer):
    """Apply softmax activation elementwise

    Parameters
    ----------
    name : str
        Used as base scope when building parameters and output
    """
    def __init__(self, name='Softmax'):
        super(Softmax, self).__init__(name=name)


class Softplus(layer.Softplus, BaseLayer):
    """Apply softplus activation elementwise

    Parameters
    ----------
    name : str
        Used as base scope when building parameters and output
    """
    def __init__(self, name='Softplus'):
        super(Softplus, self).__init__(name=name)
