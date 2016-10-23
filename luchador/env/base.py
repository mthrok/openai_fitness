from __future__ import absolute_import

from luchador.common import get_subclasses

__all__ = ['BaseEnvironment', 'get_env']


class Outcome(object):
    """Outcome when taking a step in environment

    Args:
      reward (number): Reward of transitioning the environment state
      observation: Observation of environmant state after transition
      terminal (bool): True if environment is in terminal state
      state (dict): Contains other environment-specific information
    """
    def __init__(self, reward, observation, terminal, state=None):
        self.reward = reward
        self.observation = observation
        self.terminal = terminal
        self.state = state if state else {}


class BaseEnvironment(object):
    @property
    def n_actions(self):
        raise NotImplementedError(
            '`n_actions` is not implemented for {}'.format(self.__class__)
        )

    def reset(self):
        """Reset the environment to start new episode

        Returns:
          Outcome: Outcome of resetting the state.
        """
        raise NotImplementedError(
            '`reset` method is not implemented for {}'.format(self.__class__)
        )

    def step(self, action):
        """Advance environment one step

        Returns:
          Outcome: Outcome of taking the given action
        """
        raise NotImplementedError(
            '`step` method is not implemented for {}'.format(self.__class__)
        )


def get_env(name):
    for Class in get_subclasses(BaseEnvironment):
        if Class.__name__ == name:
            return Class
    raise ValueError('Unknown Environment: {}'.format(name))
