from __future__ import division
from __future__ import absolute_import

import numpy as np

from .base import BaseAgent


class EGreedyAgent(BaseAgent):
    """Simple E-Greedy policy for stationary environment"""
    def __init__(self, epsilon, seed=None):
        self.epsilon = epsilon
        self.rng = np.random.RandomState(seed=seed)

        self.n_actions = None
        self.q_values = None
        self.n_trials = None

    def reset(self, observation):
        self.q_values = [0.0] * self.n_actions
        self.n_trials = [0.0] * self.n_actions

    def init(self, env):
        """Initialize action value and conuter"""
        self.n_actions = env.n_actions

    def observe(self, action, outcome):
        """Update the action value of based on observed outcome"""
        r, n, q = outcome.rewrad, self.n_trials[action], self.q_values[action]
        self.q_values[action] += (r - q) / (n + 1)
        self.n_trials[action] += 1

    def act(self):
        """Choose action based on e-greedy policy"""
        if self.rng.rand() < self.epsilon:
            return self.rng.randint(self.n_actions)
        else:
            return np.argmax(self.q_values)
