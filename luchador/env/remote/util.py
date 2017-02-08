"""Define utilities ([de]serialization for outcome) for remote"""
from __future__ import absolute_import

import numpy as np

from luchador.util import serialize_numpy_array, deserialize_numpy_array
from ..base import Outcome


def _serialize_state(state, compress):
    if isinstance(state, np.ndarray):
        return {
            'type': 'np.ndarray',
            'content': serialize_numpy_array(state, compress)
        }
    return {'type': 'other', 'content': state}


def _deserialize_state(obs):
    if obs['type'] == 'np.ndarray':
        return deserialize_numpy_array(obs['content'])
    return obs['content']


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
