from __future__ import division
from __future__ import absolute_import

import logging
from collections import deque

import numpy as np

import luchador

_LG = logging.getLogger(__name__)


__all__ = ['EpisodeRecorder', 'TransitionRecorder']


class EpisodeRecorder(object):
    """Class for recording a single episode

    In this class,
      - `observation` referes to an observation of environment.
      - `state` referes to a set of `observation`s combined together as a
         processing unit passed down to further processing pipeline.
      - `transition` refers to a set of (original state, action, reward,
        new state, continue flag).
    """
    def __init__(self, initial_observation):
        # So as to keep all the histories the same length all the time, fill
        # dummy action and reward at the beginning
        self.actions = [0]
        self.rewards = [0]
        self.terminals = [False]
        self.observations = [initial_observation]

    def __len__(self):
        """Return the number of actions recorded"""
        return len(self.actions) - 1

    def record(self, action, observation, reward, terminal):
        self.actions.append(action)
        self.observations.append(observation)
        self.rewards.append(reward)
        self.terminals.append(terminal)

    def _create_state(self, index, length):
        """Create state matrix from observations.

        Args:
          index (int): Index for the last observation to include in the
            resulting state.

            For performance reason, argument validity check is omitted, but
            arguments must satisfy the following condition:
              0 <= length - 1 <= index < len(self.actions)

          length (int): The number of observations included in the state

        Returns:
          NumPy NDArray: Where the length of the first axis equals to
            the given length, and the other axis corresponds to the shape
            observation.
        """
        i_start, i_end = index - length + 1, index + 1
        return self.observations[i_start:i_end]

    def _create_transition(self, index, length):
        """Create a transition (pre_state, action -> reward, post_state) from history

        Args:
          index (int) : Index of action history.

            For performance reason, argument validity check is omitted, but
            arguments must satisfy the following condition:
              0 < length <= index < len(self.actions)

          length (int): The number of observations included in the state

        Returns:
          dictionary: keys and values represent the followings
              - `action` is the action stored at given index.
              - `pre_state` is the state which contains the `n_observations`
                observations before the action is taken.
              - `reward` is the reward achieved by the taken action.
              - `post_state` is the state transitioned by taking the action at
                `pre_state`, and contains `n_observations - 1` observations
                before the action and one observation after action.
              - `terminal`: True if the `post_state` is the end of
                episode, otherwise False

            `pre_state` and `post_state` are NumPy NDArrays with same shape.
        """
        return {
            # 'index': index,  # For debug
            'pre_state': self._create_state(index-1, length),
            'action': self.actions[index],
            'reward': self.rewards[index],
            'post_state': self._create_state(index, length),
            'terminal': self.terminals[index],
        }

    def sample(self, state_length):
        index = np.random.randint(state_length, len(self.actions))
        return self._create_transition(index, state_length)


class TransitionRecorder(object):
    """Class for recording and sampling state transition"""
    def __init__(
            self, memory_size=1000000,
            state_length=4, state_width=84, state_height=84, batch_size=32,
            data_format=None):
        """Initialize TransitionRecorder class

        Args:
          state_length(int): The number of observations in state.
          n_obaservations_to_retaion(int): The number of latest observations
            to keep in memory
        """
        self.memory_size = memory_size
        self.state_length = state_length

        # EpisodeRecorder which records the current episode
        self.recorder = None
        # Set of recorders from past episodes
        self.recorders = deque()
        # Buffer for storing sampled mini-batch
        self.batch = None

        self._init_batch(width=state_width, height=state_height,
                         batch=batch_size, data_format=data_format)

    def _init_batch(self, width, height, batch, data_format):
        n, c = batch, self.state_length
        state_shape = (n, c, height, width)
        self.batch = {
            '_pre_states': np.empty(state_shape, dtype=np.uint8),
            'actions': np.empty((n,), dtype=np.uint8),
            'rewards': np.empty((n,), dtype=np.float32),
            '_post_states': np.empty(state_shape, dtype=np.uint8),
            'terminals': np.empty((n,), dtype=np.bool_),
        }

        data_format = data_format or luchador.get_nn_conv_format()
        self.batch['pre_states'] = (
            self.batch['_pre_states'] if data_format == 'NCHW' else
            self.batch['_pre_states'].transpose((0, 2, 3, 1)))
        self.batch['post_states'] = (
            self.batch['_post_states'] if data_format == 'NCHW' else
            self.batch['_post_states'].transpose((0, 2, 3, 1)))

    def __len__(self):
        """Return the total number of actions stored"""
        return sum(len(r) for r in self.recorders)

    def reset(self, initial_observation):
        """Initialize recorder with new EpisodeRecorder"""
        self.recorder = EpisodeRecorder(initial_observation)

    def record(self, action, observation, reward, terminal):
        self.recorder.record(
            action=action, observation=observation,
            reward=reward, terminal=terminal)

    def truncate(self):
        """Retain only recorders containing the latest records"""
        self.recorders.append(self.recorder)
        self.recorder = None

        n_records = len(self)
        n_recorders = len(self.recorders)
        while n_recorders > 1 and n_records > self.memory_size:
            poped = self.recorders.popleft()
            n_records -= len(poped)
            n_recorders -= 1

        _LG.debug('  After truncate:')
        _LG.debug('  # records  : %s', n_records)
        _LG.debug('  # recorders: %s', n_recorders)

    def is_ready(self):
        """True if the current episode recorder has sufficient records"""
        return len(self.recorder) > self.state_length

    def get_current_state(self):
        """Get the latest state"""
        n_obs = self.state_length
        self.batch['_pre_states'][0] = self.recorder.observations[-n_obs:]
        return self.batch['pre_states'][0:1]

    def sample(self, n_samples):
        """Sample state transitions into given batch"""
        i = 0
        while i < n_samples:
            i_recorder = np.random.randint(len(self.recorders))
            recorder = self.recorders[i_recorder]
            if len(recorder) < self.state_length:
                continue
            transition = recorder.sample(self.state_length)
            # self.batch['index'][i] = transition['index']  # For debug
            self.batch['_pre_states'][i] = transition['pre_state']
            self.batch['actions'][i] = transition['action']
            self.batch['rewards'][i] = transition['reward']
            self.batch['_post_states'][i] = transition['post_state']
            self.batch['terminals'][i] = transition['terminal']
            i += 1
        return self.batch
