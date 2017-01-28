"""Module to implement funcamental RL concepts"""
from __future__ import division
from __future__ import absolute_import

import numpy as np
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


class WienerNoise(object):
    """Generate noise with Wiener process, i.e. gaussian noise

    Parameters
    ----------
    shape : tuple
        Shape of the noise to generate

    delta, dt : float
        Parameters to controll the process.
        ``dt`` represents the time difference between calls to :func:`sample`
        ``delta`` represents standard deviation

        .. math::

            x \\sim \\mathcal{N} (\\mu=0, \\sigma=delta^2 * dt)

    seed
        Seed for random generator
    """
    def __init__(self, shape, delta, dt, seed=None):
        self.delta = delta
        self.dt = dt
        self.shape = shape

        self._sigma = self.dt * (self.delta ** 2)
        self._rng = RandomState(seed=seed)

    def reset(self):
        """Reset internal state. Do nothing for WienerNoise"""
        pass

    def sample(self):
        """Sample noise

        See :func:`WienerNoise` constructor for the value sampled.
        """
        return self._sigma * self._rng.randn(*self.shape)


class OUNoise(WienerNoise):
    """Generate temporally correlated noise with Ornstein-Uhlenbeck process

    Parameters
    ----------
    shape : tuple
        Shape of the noise to generate

    mu, sigma, theta : float
        Parameters to controll the process.

        .. math::

            x &\\leftarrow \\mu

            dx &\\sim \\theta * (\\mu - x) +
            \\sigma * \\mathcal{N} (\\mu=0, \\sigma=1)

            x &\\leftarrow x + dx

    seed
        Seed for random generator
    """
    def __init__(self, shape, mu, sigma, theta, seed=None):
        super(OUNoise, self).__init__(shape=shape, delta=1, dt=1, seed=seed)
        self.mu = mu
        self.sigma = sigma
        self.theta = theta

        self._state = None
        self.reset()

    def reset(self):
        """Reset the internal state to :math:`mu`"""
        self._state = self.mu * np.ones(self.shape)

    def sample(self):
        """Sample noise

        See :func:`OUNoise` constructor for the value sampled.
        """
        base = super(OUNoise, self).sample()
        diff = self.theta * (self.mu - self._state) + self.sigma * base
        self._state += diff
        return np.copy(self._state)
