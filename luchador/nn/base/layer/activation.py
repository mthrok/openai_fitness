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
    """Apply rectified linear activation elementwise"""
    def __init__(self):
        super(BaseReLU, self).__init__()


class BaseSigmoid(BaseLayer):
    """Apply sigmoid activation elementwise"""
    def __init__(self):
        super(BaseSigmoid, self).__init__()


class BaseTanh(BaseLayer):
    """Apply tanh activation elementwise"""
    def __init__(self):
        super(BaseTanh, self).__init__()


class BaseSoftmax(BaseLayer):
    """Apply softmax activation elementwise"""
    def __init__(self):
        super(BaseSoftmax, self).__init__()


class BaseSoftplus(BaseLayer):
    """Apply softplus activation elementwise"""
    def __init__(self):
        super(BaseSoftplus, self).__init__()
