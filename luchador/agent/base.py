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
    def observe(self, action, outcome):
        """Observe the action and it's outcome.

        Parameters
        ----------
        action : int
            The action that this agent previously took.

        oucome : Outome
            Outcome of taking the action
        """
        pass

    @abc.abstractmethod
    def act(self, observation):
        """Choose action.

        Parameters
        ----------
        observation
            Current observation of environment.
        """
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

    def observe(self, action, outcome):
        pass

    def act(self, _):
        return 0


_AGENT_MODULE_MAPPING = {
    'DQNAgent': 'dqn',
}


def get_agent(name):
    """Retrieve Agent class by name

    Parameters
    ----------
    name : str
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
    if name in _AGENT_MODULE_MAPPING:
        module = 'luchador.agent.{:s}'.format(_AGENT_MODULE_MAPPING[name])
        importlib.import_module(module)

    for class_ in luchador.util.get_subclasses(BaseAgent):
        if class_.__name__ == name:
            return class_
    raise ValueError('Unknown Agent: {}'.format(name))
