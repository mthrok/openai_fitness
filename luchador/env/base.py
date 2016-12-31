"""Define common interface for Environment"""
from __future__ import absolute_import

import abc
import importlib

import numpy as np

import luchador


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
        self.state = state or {}


def _serialize_observation(obs):
    if isinstance(obs, np.ndarray):
        return {
            'type': 'np.ndarray',
            'obj': {
                'shape': obs.shape,
                'dtype': str(obs.dtype),
                'data': obs.tostring().encode('base64'),
            }
        }
    return {
        'type': 'other',
        'obj': obs
    }


def _deserialize_observation(obs):
    if obs['type'] == 'np.ndarray':
        data, dtype = obs['obj']['data'], obs['obj']['dtype']
        data = np.fromstring(data.decode('base64'), dtype=dtype)
        return data.reshape(obs['obj']['shape'])
    return obs['obj']


def serialize_outcome(outcome):
    """Serialize observation to JSON

    Returns
    -------
    dict
        Outcome components in dictionary format
    """
    return {
        'reward': outcome.reward,
        'observation': _serialize_observation(outcome.observation),
        'terminal': outcome.terminal,
        'state': outcome.state
    }


def deserialize_outcome(obj):
    """Deserialize Outcome from JSON

    Parameters
    ----------
    obj : dict
        Outcome instance serialized with :any:`serialize_outcome`
    """
    obs = _deserialize_observation(obj['observation'])
    return Outcome(obj['reward'], obs, obj['terminal'], obj['state'])


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
}


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
    # Some environments depend on dynamic library, and
    # importing all of them at global initialization leads to TLS Error on
    # Ubuntu 14.04.
    # TLS Error is avoidable only by upgrading underlying libc version, which
    # is not easy. So we import such environments on-demand.
    if name in _ENVIRONMENT_MODULE_MAPPING:
        module = 'luchador.env.{:s}'.format(_ENVIRONMENT_MODULE_MAPPING[name])
        importlib.import_module(module)

    for class_ in luchador.common.get_subclasses(BaseEnvironment):
        if class_.__name__ == name:
            return class_

    raise ValueError('Unknown Environment: {}'.format(name))
