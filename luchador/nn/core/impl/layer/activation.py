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
    scope : str
        Used as base scope when building parameters and output
    """
    def __init__(self, scope='ReLU'):
        super(ReLU, self).__init__(scope=scope)


class LeakyReLU(layer.LeakyReLU, BaseLayer):
    """Apply rectified linear activation elementwise

    Parameters
    ----------
    alpha : float
        Initial value of slope for negative region.

    train : bool
        When True, ``alpha`` variable becomes trainable variable.

    scope : str
        Used as base scope when building parameters and output
    """
    def __init__(self, alpha, train=False, scope='LeakyReLU'):
        super(LeakyReLU, self).__init__(scope=scope, train=train, alpha=alpha)


class Sigmoid(layer.Sigmoid, BaseLayer):
    """Apply sigmoid activation elementwise

    Parameters
    ----------
    scope : str
        Used as base scope when building parameters and output
    """
    def __init__(self, scope='Sigmoid'):
        super(Sigmoid, self).__init__(scope=scope)


class Tanh(layer.Tanh, BaseLayer):
    """Apply tanh activation elementwise

    Parameters
    ----------
    scope : str
        Used as base scope when building parameters and output
    """
    def __init__(self, scope='Tanh'):
        super(Tanh, self).__init__(scope=scope)


class Softmax(layer.Softmax, BaseLayer):
    """Apply softmax activation elementwise

    Parameters
    ----------
    scope : str
        Used as base scope when building parameters and output
    """
    def __init__(self, scope='Softmax'):
        super(Softmax, self).__init__(scope=scope)


class Softplus(layer.Softplus, BaseLayer):
    """Apply softplus activation elementwise

    Parameters
    ----------
    scope : str
        Used as base scope when building parameters and output
    """
    def __init__(self, scope='Softplus'):
        super(Softplus, self).__init__(scope=scope)
