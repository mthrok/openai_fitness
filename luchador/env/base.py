"""Define common interface for Environment"""
from __future__ import absolute_import

import abc
import importlib
from collections import namedtuple

from luchador.util import fetch_subclasses

_Outcome = namedtuple('_Outcome', ('reward', 'state', 'terminal', 'info'))


class Outcome(_Outcome):
    """Outcome when taking a step in environment

    Parameters
    ----------
    reward : number
        Reward of transitioning the environment state
    observation
        Observation of environmant state after transition
    terminal : bool
        True if environment is in terminal state
    info : dict
        Other environment-specific information
    """
    def __new__(cls, reward, state, terminal, info=None):
        return super(Outcome, cls).__new__(
            cls, reward=reward, state=state, terminal=terminal, info=info)


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


_ENVIRONMENT_MODULE_MAPPING = {
    'ALEEnvironment': 'ale',
    'FlappyBird': 'flappy_bird',
    'CartPole': 'cart_pole',
    'RPiRover': 'rpi_rover',
}


def get_env(typename):
    """Retrieve Environment class by typename

    Parameters
    ----------
    typename : str
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
    # Some environments depend on dynamic library, and
    # importing all of them at global initialization leads to TLS Error on
    # Ubuntu 14.04.
    # TLS Error is avoidable only by upgrading underlying libc version, which
    # is not easy. So we import such environments on-demand.
    if typename in _ENVIRONMENT_MODULE_MAPPING:
        module = 'luchador.env.{:s}'.format(
            _ENVIRONMENT_MODULE_MAPPING[typename])
        importlib.import_module(module)

    for class_ in fetch_subclasses(BaseEnvironment):
        if class_.__name__ == typename:
            return class_

    raise ValueError('Unknown Environment: {}'.format(typename))
