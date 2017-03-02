"""Module for implementing random source"""
from __future__ import absolute_import

from ...base import BaseLayer
from ...backend import layer
# pylint: disable=abstract-method


def _validate_mode(mode):
    if mode.lower() not in ['add', 'multiply']:
        raise ValueError('`mode` must be either "add" or "multiply"')


class NormalNoise(layer.NormalNoise, BaseLayer):
    """Add random values from a normal distribution to input

    Parameters
    ----------
    mean : float
        The mean of the normal distribution.

    stddev : float
        The standard deviation of the normal distribution.

    mode : str
        'add' or 'multiply'. Determine if noise value is added to
        the input or multiply the input elementwise.

    seed : A Python integer.
        Random seed for the distribution.

    name : str
        Scope for the output tensor.
    """
    def __init__(
            self, mean=0.0, std=1.0, mode='add',
            seed=None, name='NormalNoise'):
        super(NormalNoise, self).__init__(
            mean=mean, std=std, mode=mode, seed=seed, name=name)
        self._rng = None

    def _validate_args(self, mode, **_):
        _validate_mode(mode)


class UniformNoise(layer.UniformNoise, BaseLayer):
    """Add random values from a uniform distribution to input

    Parameters
    ----------
    low : float
        The lower bound of the uniform distribution.

    high : float
        The Upper bound of the uniform distribution.

    mode : str
        'add' or 'multiply'. Determine if noise value is added to
        the input or multiply the input elementwise.

    seed : A Python integer.
        Random seed for the distribution.

    name : str
        Scope for the output tensor.
    """
    def __init__(
            self, low=0.0, high=1.0, mode='add',
            seed=None, name='UniformNoise'):
        super(UniformNoise, self).__init__(
            low=low, high=high, mode=mode, seed=seed, name=name)
        self._rng = None

    def _validate_args(self, mode, **_):
        _validate_mode(mode)
