"""Illustrates how to implement custom agent"""
from luchador.agent import BaseAgent

import numpy as np


class MyRandomAgent(BaseAgent):
    """Illustrates how to implement custom agent"""
    def __init__(self):
        self.n_actions = 0

    def init(self, env):
        self.n_actions = env.n_actions

    def reset(self, observation):
        pass

    def observe(self, action, outcome):
        pass

    def act(self):
        return np.random.randint(self.n_actions)
