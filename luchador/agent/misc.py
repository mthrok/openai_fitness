"""Module to implement funcamental RL concepts"""
from __future__ import division
from __future__ import absolute_import

from numpy.random import RandomState


class EGreedy(object):
    """Linearly anneal exploration probability

    Parameters
    ----------
    epsilon_init : float
        Initial epsilon

    epsilon_term : float
        Terminal epsilon

    duration : int
        Duration to anneal epsilon

    method : str
        Annealing method. Only 'linear' is supported
    """
    def __init__(
            self, epsilon_init, epsilon_term, duration,
            method='linear', seed=None):
        if method not in ['linear']:
            raise ValueError('method must be one of ["linear"]')

        self.epsilon_init = epsilon_init
        self.epsilon_term = epsilon_term
        self.duration = duration
        self.method = method
        self.seed = seed

        self.count = 0
        self._rng = RandomState(seed=seed)

    @property
    def epsilon(self):
        """Current epsilon value"""
        if self.count >= self.duration:
            return self.epsilon_term

        if self.method == 'linear':
            diff = (self.epsilon_term - self.epsilon_init) / self.duration
            return self.epsilon_init + self.count * diff

    def act_random(self):
        """Decide weather to take random action"""
        ret = self._rng.rand() < self.epsilon
        self.count += 1
        return ret
