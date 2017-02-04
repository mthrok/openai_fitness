"""Define agent interface"""

from __future__ import absolute_import
import abc
import importlib

import luchador.util


__all__ = ['BaseAgent', 'NoOpAgent', 'get_agent']


class BaseAgent(object):
    """Define interface for Agent class"""
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def init(self, env):
        """Inittialize agent under the given env

        This method serves to retrieve environment-specific information,
        such as the number of action the agent can take, from environment.

        Parameters
        ----------
        env : Environment
            The environment agent works on.
        """
        pass

    @abc.abstractmethod
    def learn(self, state0, action, reward, state1, terminal, info):
        """Observe the action and it's outcome.

        Parameters
        ----------
        state0
            Environment state before taking action
        action
            The action taken
        reward : float
            Reward acquired by taking action
        state1
            Environment state after taking action
        terminal : Bool
            True if state1 is terminal state
        info
            Supplemental information of environment
        """
        pass

    @abc.abstractmethod
    def act(self):
        """Choose action."""
        pass

    @abc.abstractmethod
    def reset(self, observation):
        """Reset agent with the initial state of the environment.

        Parameters
        ----------
        observation
            Observation made when environment is reset
        """
        pass

    def perform_post_episode_task(self, stats):
        """Perform post episode task"""
        pass


class NoOpAgent(BaseAgent):
    """Agent does nothing"""
    def __init__(self):
        super(NoOpAgent, self).__init__()

    def init(self, env):
        pass

    def reset(self, observation):
        pass

    def learn(self, state0, action, reward, state1, terminal, info=None):
        pass

    def act(self):
        return 0

    def __str__(self):
        return 'NoOpAgent'


_AGENT_MODULE_MAPPING = {
    'DQNAgent': 'dqn',
}


def get_agent(typename):
    """Retrieve Agent class by name

    Parameters
    ----------
    typename : str
        Name of Agent to retrieve

    Returns
    -------
    type
        Agent type found

    Raises
    ------
    ValueError
        When Agent with the given name is not found
    """
    if typename in _AGENT_MODULE_MAPPING:
        module = 'luchador.agent.{:s}'.format(_AGENT_MODULE_MAPPING[typename])
        importlib.import_module(module)

    for class_ in luchador.util.get_subclasses(BaseAgent):
        if class_.__name__ == typename:
            return class_
    raise ValueError('Unknown Agent: {}'.format(typename))
