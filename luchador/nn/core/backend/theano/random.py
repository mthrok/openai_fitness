from __future__ import absolute_import

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from ...base.wrapper import BaseRandomSource

__all__ = ['NormalRandom', 'UniformRandom']


class NormalRandom(BaseRandomSource):
    def __init__(self, mean=0.0, std=1.0, seed=None, name=None):
        self.mean = mean
        self.std = std
        self.name = name
        self._rng = RandomStreams(seed=seed or 123456)

    def sample(self, shape, dtype):
        return self._rng.normal(
            size=shape, avg=self.mean, std=self.std, dtype=dtype)


class UniformRandom(BaseRandomSource):
    def __init__(self, low=0.0, high=1.0, seed=None, name=None):
        self.low = low
        self.high = high
        self.name = name
        self._rng = RandomStreams(seed=seed or 123456)

    def sample(self, shape, dtype):
        return self._rng.uniform(
            size=shape, low=self.low, high=self.high, dtype=dtype)
