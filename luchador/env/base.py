"""Define common interface for Environment"""

from __future__ import absolute_import

import abc

from luchador import common

__all__ = ['BaseEnvironment', 'get_env']


class Outcome(object):
    """Outcome when taking a step in environment

    Parameters
    ----------
    reward : number
        Reward of transitioning the environment state
    observation
        Observation of environmant state after transition
    terminal : bool
        True if environment is in terminal state
    state : dict
        Other environment-specific information
    """
    def __init__(self, reward, observation, terminal, state=None):
        self.reward = reward
        self.observation = observation
        self.terminal = terminal
        self.state = state if state else {}


class BaseEnvironment(object):
    """Define common interface for environment"""
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def n_actions(self):
        """Return the number of actions agent can take"""
        pass

    @abc.abstractmethod
    def reset(self):
        """Reset the environment to start new episode

        Returns
        -------
        Outcome
            Outcome of resetting the state.
        """
        pass

    @abc.abstractmethod
    def step(self, action):
        """Advance environment one step

        Parameters
        ----------
        action : int
            Action taken by an agent

        Returns
        -------
        Outcome
            Outcome of taking the given action
        """
        pass


def get_env(name):
    """Retrieve Environment class by name

    Parameters
    ----------
    name : str
        Name of Environment to retrieve

    Returns
    -------
    type
        Environment type found

    Raises
    ------
    ValueError
        When Environment with the given name is not found
    """
    for class_ in common.get_subclasses(BaseEnvironment):
        if class_.__name__ == name:
            return class_
    raise ValueError('Unknown Environment: {}'.format(name))
