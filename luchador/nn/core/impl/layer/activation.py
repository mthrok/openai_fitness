"""Define common interface for Layer classes"""
from __future__ import division
from __future__ import absolute_import

from ...base.layer import BaseLayer
from ...backend import layer

__all__ = ['ReLU', 'Sigmoid', 'Tanh', 'Softmax', 'Softplus']


class ReLU(layer.ReLU, BaseLayer):
    """Apply rectified linear activation elementwise

    Parameters
    ----------
    name : str
        Used as base scope when building parameters and output
    """
    def __init__(self, name='ReLU'):
        super(ReLU, self).__init__(name=name)


class Sigmoid(layer.ReLU, BaseLayer):
    """Apply sigmoid activation elementwise

    Parameters
    ----------
    name : str
        Used as base scope when building parameters and output
    """
    def __init__(self, name='Sigmoid'):
        super(Sigmoid, self).__init__(name=name)


class Tanh(layer.ReLU, BaseLayer):
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
