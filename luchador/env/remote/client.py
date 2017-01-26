"""Module for facilitating the access to remote Environment server"""
from __future__ import absolute_import

import json
import requests

import luchador.env.base as base_env


def _extract_outcome(response):
    return base_env.deserialize_outcome(response.json())


class RemoteEnv(base_env.BaseEnvironment):
    """Simple interface to access remote Environment server

    Parameters
    ----------
    port : str
        Port number of the server

    host : str
        Host address of the server
    """
    def __init__(self, port, host='0.0.0.0'):
        self.port = port
        self.host = host

    @property
    def _url(self):
        return 'http://{}:{}'.format(self.host, self.port)

    def info(self):
        """Get environment infor by calling ``/info`` route

        Returns
        -------
        dict
            Environment information
        """
        res = requests.post('{}/info'.format(self._url))
        return json.loads(res.json()['environment'])

    @property
    def n_actions(self):
        """Get #valid actions by calling ``/n_actions`` route

        Returns
        -------
        int
            The number of valid actions for the Environment
        """
        res = requests.post('{}/n_actions'.format(self._url))
        return res.json()['n_actions']

    def outcome(self):
        """Get the last outcome by calling ``/outcome`` route

        Returns
        -------
        Outcome
            Outcome from the last transition
        """
        res = requests.post('{}/outcome'.format(self._url))
        return _extract_outcome(res)

    def reset(self):
        """Reset the env by calling ``/reset`` route

        Returns
        -------
        Outcome
            Outcome of resetting the Environment
        """
        res = requests.post('{}/reset'.format(self._url))
        return _extract_outcome(res)

    def step(self, action):
        """Advance the env by calling ``/step`` route

        Returns
        -------
        Outcome
            Outcome of taking the step
        """
        url = '{}/step'.format(self._url)
        res = requests.post(url, json={'action': action})
        return _extract_outcome(res)

    def kill(self):
        """Kill the env by calling ``/kill`` route

        Returns
        -------
        Bool
            True if server is correctly stopped else False
        """
        res = requests.post('{}/kill'.format(self._url))
        return res.json()['result'] == 'success'

    def __str__(self):
        return 'RemoteEnv: Host: {}, Port: {}'.format(self.host, self.port)
