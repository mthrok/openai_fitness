from __future__ import division
from __future__ import absolute_import

import numpy as np

from .base import BaseAgent


class EGreedyAgent(BaseAgent):
    """Simple E-Greedy policy for stationary environment

    Parameters
    ----------
    epsolon : float
        The probability to take random action.

    step_size : 'average' or float
        Parameter to adjust how action value is estimated from the series of
        observations. When 'average', estimated action value is simply the mean
        of all the observed rewards for the action. When float, estimation is
        updated with weighted sum over current estimation and newly observed
        reward value.

    seed : int
        Random seed
    """
    def __init__(self, epsilon, step_size='average', seed=None):
        self.epsilon = epsilon
        self.step_size = step_size
        self.rng = np.random.RandomState(seed=seed)

        self.n_actions = None
        self.q_values = None
        self.n_trials = None

    def reset(self, observation):
        self.q_values = [0.0] * self.n_actions
        self.n_trials = [0.0] * self.n_actions

    def init(self, env):
        self.n_actions = env.n_actions

    def observe(self, action, outcome):
        """Update the action value estimation based on observed outcome"""
        r, n, q = outcome.reward, self.n_trials[action], self.q_values[action]
        alpha = 1 / (n + 1) if self.step_size == 'average' else self.step_size
        self.q_values[action] += (r - q) * alpha
        self.n_trials[action] += 1

    def act(self):
        """Choose action based on e-greedy policy"""
        if self.rng.rand() < self.epsilon:
            return self.rng.randint(self.n_actions)
        else:
            return np.argmax(self.q_values)
