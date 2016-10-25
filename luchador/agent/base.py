from __future__ import absolute_import
import abc

from luchador import common

__all__ = ['BaseAgent', 'NoOpAgent', 'get_agent']


class BaseAgent(object):
    __metaclass__ = abc.ABCMeta

    def init(self, env):
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
    def __init__(self):
        super(NoOpAgent, self).__init__()

    def init(self, env):
        pass

    def reset(self, observation):
        pass

    def observe(self, action, outcome):
        pass

    def act(self):
        return 0


def get_agent(name):
    for class_ in common.get_subclasses(BaseAgent):
        if class_.__name__ == name:
            return class_
    raise ValueError('Unknown Agent: {}'.format(name))
