"""Define common interface for Layer classes"""
from __future__ import division
from __future__ import absolute_import

from .base import BaseLayer

__all__ = [
    'BaseReLU', 'BaseSigmoid', 'BaseTanh',
    'BaseSoftmax', 'BaseSoftplus',
]

# pylint: disable=abstract-method


class BaseReLU(BaseLayer):
    """Apply rectified linear activation elementwise

    Parameters
    ----------
    name : str
        Used as base scope when building parameters and output
    """
    def __init__(self, name='ReLU'):
        super(BaseReLU, self).__init__(name=name)


class BaseSigmoid(BaseLayer):
    """Apply sigmoid activation elementwise

    Parameters
    ----------
    name : str
        Used as base scope when building parameters and output
    """
    def __init__(self, name='Sigmoid'):
        super(BaseSigmoid, self).__init__(name=name)


class BaseTanh(BaseLayer):
    """Apply tanh activation elementwise

    Parameters
    ----------
    name : str
        Used as base scope when building parameters and output
    """
    def __init__(self, name='Tanh'):
        super(BaseTanh, self).__init__(name=name)


class BaseSoftmax(BaseLayer):
    """Apply softmax activation elementwise

    Parameters
    ----------
    name : str
        Used as base scope when building parameters and output
    """
    def __init__(self, name='Softmax'):
        super(BaseSoftmax, self).__init__(name=name)


class BaseSoftplus(BaseLayer):
    """Apply softplus activation elementwise

    Parameters
    ----------
    name : str
        Used as base scope when building parameters and output
    """
    def __init__(self, name='Softplus'):
        super(BaseSoftplus, self).__init__(name=name)
