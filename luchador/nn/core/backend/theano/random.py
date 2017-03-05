"""Module foe implementing random source"""
from __future__ import absolute_import

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

__all__ = ['NormalRandom', 'UniformRandom']
# pylint: disable=no-member


class NormalRandom(object):
    """Implements normal random sampling in Tensorflow"""
    def __init__(self):
        self._rng = RandomStreams(seed=self.seed or 123456)

    def _sample(self, shape, dtype):
        return self._rng.normal(
            size=shape, avg=self.mean, std=self.std, dtype=dtype)


class UniformRandom(object):
    """Implements uniform random sampling in Tensorflow"""
    def __init__(self):
        self._rng = RandomStreams(seed=self.seed or 123456)

    def _sample(self, shape, dtype):
        return self._rng.uniform(
            size=shape, low=self.low, high=self.high, dtype=dtype)
