from __future__ import absolute_import

import numpy as np

from .base import BaseEnvironment, Outcome


class _Normal(object):
    def __init__(self, mean, variance, rng):
        self.rng = rng
        self.mean = mean
        self.stddev = np.sqrt(variance)

    def sample(self):
        """Sample from Normal distribution"""
        return self.mean + self.stddev * self.rng.randn()


def _build_distributions(n_dists, rng):
    return [
        _Normal(mean=rng.randn(), variance=1, rng=rng)
        for _ in range(n_dists)]


class StationaryBandit(BaseEnvironment):
    """N-armed bandit problem

    Parameters
    ----------
    n_arms : int
        The number of arms of bandit
    """
    def __init__(self, n_arms, seed=None):
        self.n_arms = n_arms
        self.rng = np.random.RandomState(seed=seed)

        self.distributions = None

    @property
    def n_actions(self):
        return self.n_arms

    def reset(self):
        self.distributions = _build_distributions(self.n_arms, self.rng)
        return Outcome(reward=0, observation=None, terminal=False)

    def step(self, n):
        reward = self.distributions[n].sample()
        return Outcome(reward=reward, observation=None, terminal=False)
