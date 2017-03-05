"""Implement wrapper classes"""
from __future__ import absolute_import

from ..base.wrapper import BaseRandomSource
from ..backend import random

# pylint: disable=invalid-name


__all__ = ['NormalRandom', 'UniformRandom']


class NormalRandom(random.NormalRandom, BaseRandomSource):
    """Generate gaussian random values

    Parameters
    ----------
    mean : float
        Mean of sampling distribution
    std : float
        Standard deviation of sampling distribution
    seed : int
        Seed for random generator.
    """
    def __init__(self, mean=0.0, std=1.0, seed=None):
        self.mean = mean
        self.std = std
        self.seed = seed
        super(NormalRandom, self).__init__()


class UniformRandom(random.UniformRandom, BaseRandomSource):
    """Generate uniform random values

    Parameters
    ----------
    low, high : float
        Lower/upper bound of distribution
    seed : int
        Seed for random generator.
    """
    def __init__(self, low=0.0, high=1.0, seed=None):
        self.low = low
        self.high = high
        self.seed = seed
        super(UniformRandom, self).__init__()
