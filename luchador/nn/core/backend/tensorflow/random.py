from __future__ import absolute_import

import tensorflow as tf

from ...base.wrapper import BaseRandomSource

__all__ = ['NormalRandom', 'UniformRandom']


class NormalRandom(BaseRandomSource):
    def __init__(self, mean=0.0, std=1.0, seed=None, name=None):
        self.mean = mean
        self.std = std
        self.name = name
        self.seed = seed

    def sample(self, shape, dtype):
        return tf.random_normal(
            shape=shape, mean=self.mean,
            stddev=self.std, dtype=dtype, seed=self.seed)


class UniformRandom(BaseRandomSource):
    def __init__(self, low=0.0, high=1.0, seed=None, name=None):
        self.low = low
        self.high = high
        self.name = name
        self.seed = seed

    def sample(self, shape, dtype):
        return tf.random_uniform(
            shape=shape, minval=self.low, maxval=self.high,
            dtype=dtype, seed=self.seed)
