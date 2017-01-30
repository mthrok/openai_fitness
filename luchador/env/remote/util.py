"""Define utilities ([de]serialization for outcome) for remote"""
from __future__ import absolute_import

import zlib

import numpy as np

from ..base import Outcome


def _serialize_state(state, compress):
    if isinstance(state, np.ndarray):
        data = state.tostring()
        if compress:
            data = zlib.compress(data)
        return {
            'type': 'np.ndarray',
            'obj': {
                'shape': state.shape,
                'dtype': str(state.dtype),
                'data': data.encode('base64'),
                'compressed': compress,
            }
        }
    return {
        'type': 'other',
        'obj': state
    }


def _deserialize_state(obs):
    if obs['type'] == 'np.ndarray':
        data, dtype = obs['obj']['data'], obs['obj']['dtype']
        data = data.decode('base64')
        if obs['obj']['compressed']:
            data = zlib.decompress(data)
        data = np.fromstring(data, dtype=dtype)
        return data.reshape(obs['obj']['shape'])
    return obs['obj']


def serialize_outcome(outcome, compress=True):
    """Serialize observation to JSON

    Parameters
    ----------
    outcome : Outcome
        Outcome object to serialize

    compress : bool
        If state is NumPy NDArray and compress = True, value is compressed
        after serialization.

    Returns
    -------
    dict
        Outcome components in dictionary format
    """
    return {
        'reward': outcome.reward,
        'state': _serialize_state(outcome.state, compress),
        'terminal': outcome.terminal,
        'info': outcome.info
    }


def deserialize_outcome(obj):
    """Deserialize Outcome from JSON

    Parameters
    ----------
    obj : dict
        Outcome instance serialized with :any:`serialize_outcome`
    """
    obs = _deserialize_state(obj['state'])
    return Outcome(obj['reward'], obs, obj['terminal'], obj['info'])
