"""Module foe implementing random source"""
from __future__ import absolute_import

import tensorflow as tf

__all__ = ['NormalRandom', 'UniformRandom']
# pylint: disable=no-member


class NormalRandom(object):
    """Implements normal random sampling in Tensorflow"""
    def _sample(self, shape, dtype):
        return tf.random_normal(
            shape=shape, mean=self.mean,
            stddev=self.std, dtype=dtype, seed=self.seed)


class UniformRandom(object):
    """Implements uniform random sampling in Tensorflow"""
    def _sample(self, shape, dtype):
        return tf.random_uniform(
            shape=shape, minval=self.low, maxval=self.high,
            dtype=dtype, seed=self.seed)
