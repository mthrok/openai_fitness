from __future__ import absolute_import

from numpy.random import RandomState


_RANDOM_STATE = RandomState(seed=None)


def set_random_seed(seed):
    global _RANDOM_STATE
    _RANDOM_STATE = RandomState(seed=seed)


def get_rng():
    return _RANDOM_STATE
