from __future__ import absolute_import

import logging

import numpy as np

from .base import BaseAgent

__all__ = ['RandomAgent']

_LG = logging.getLogger(__name__)


class RandomAgent(BaseAgent):
    def __init__(self):
        super(RandomAgent, self).__init__()

    def set_env_info(self, env):
        self.n_actions = env.n_actions

    def reset(self, observation):
        pass

    def observe(self, action, outcome):
        pass

    def act(self):
        return np.random.randint(self.n_actions)

    def __repr__(self):
        return (
            '[RandomAgent]'
            '    n_actions: {}\n'.format(self.n_actions)
        )
