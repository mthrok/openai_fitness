from __future__ import absolute_import

from luchador.common import get_subclasses

__all__ = ['BaseAgent', 'NoOpAgent', 'get_agent']


class BaseAgent(object):
    def init(self, env):
        pass

    def observe(self, action, outcome):
        """Observe the action and it's outcome.

        Args:
          action: The action that this agent previously took.
          oucome (Outome): Outcome of taking the action
        """
        raise NotImplementedError('observe method is not implemented.')

    def act(self):
        """Choose action. Must be implemented in subclass."""
        raise NotImplementedError('act method is not implemented.')

    def reset(self, observation):
        """Reset agent with the initial state of the environment."""
        raise NotImplementedError('reset method is not implemented.')

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
    for class_ in get_subclasses(BaseAgent):
        if class_.__name__ == name:
            return class_
    raise ValueError('Unknown Agent: {}'.format(name))
