from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import time
import unittest

import numpy as np

from tests.unit.fixture import create_image

from luchador.agent.recorder import (
    EpisodeRecorder,
    TransitionRecorder,
)

base_state = create_image(height=3, width=5, channel=None)


def _make_episode_recorder(stack, transition=2):
    """Create Episoderecorder with the same configuration/initial data
    as the original implementation"""
    er = EpisodeRecorder(
        buffer_configs={
            'state': {'stack': stack, 'transition': transition},
            'action': {'stack': None, 'transition': None},
            'reward': {'stack': None, 'transition': None},
            'terminal': {'stack': None, 'transition': None},
        },
        initial_data={
            'state': 0*base_state,
        }
    )
    return er


def _create_transition_recorder(memory_size, batch_size=32, stack=4):
    return TransitionRecorder(
        memory_size=memory_size,
        buffer_config={
            'state': {
                'stack': stack, 'transition': 2,
                'shape': base_state.shape, 'dtype': 'uint8'
            },
            'action': {
                'stack': None, 'transition': None,
                'dtype': 'uint8'
            },
            'reward': {'dtype': 'float32'},
            'terminal': {'dtype': 'bool'},
        },
        batch_size=batch_size,
    )

# pylint: disable=protected-access


class TestEpisodeRecorder(unittest.TestCase):
    def test_record(self):
        """EpisodeRecorder correctly records histories"""
        er = _make_episode_recorder(stack=11)

        n_actions = 5 * er.max_stack
        for i in range(n_actions):
            action, reward, terminal = i, 2 * i, bool(i % 3)
            state = (i+1) * base_state
            er.put({
                'action': action, 'reward': reward,
                'state': state, 'terminal': terminal
            })

        for i in range(n_actions):
            expected = i * base_state
            found = er.buffers['state'].data[i]
            self.assertTrue(
                (found == expected).all(),
                'Observation is not correctly recorded. '
                'Expected: {}, Found: {}.'.format(expected, found))

        for i in range(n_actions):
            expected = i
            found = er.buffers['action'].data[i]
            self.assertEqual(
                expected, found,
                'Action is not correctly recorded. '
                'Expected: {}, Found: {}.'.format(expected, found))
            expected = 2 * i
            found = er.buffers['reward'].data[i]
            self.assertEqual(
                expected, found,
                'Reward is not correctly recorded. '
                'Expected: {}, Found: {}.'.format(expected, found))
            expected = bool(i % 3)
            found = er.buffers['terminal'].data[i]
            self.assertEqual(
                expected, found,
                'Terminal is not correctly recorded. '
                'Expected: {}, Found: {}.'.format(expected, found))

    def test_create_state(self):
        """EpisodeRecorder correctly creates state from observations"""
        er = _make_episode_recorder(stack=5)

        n_actions = 3 * er.max_stack
        for i in range(n_actions):
            action, reward, terminal = i, 2 * i, bool(i % 3)
            state = (i+1) * base_state
            er.put({
                'action': action, 'reward': reward,
                'state': state, 'terminal': terminal,
            })

        for i in range(er.max_stack-1, n_actions):
            transition = er.get(i)
            state0 = transition['state'][0]
            state1 = transition['state'][1]
            j_start = i - er.max_stack + 1
            for j in range(er.max_stack):
                expected = (j_start + j) * base_state
                found = state0[j]
                self.assertTrue(
                    (found == expected).all(),
                    'State is not properly created for index {} slice {}. '
                    'Expected: {}, Found: {}'.format(i, j, expected, found))
                expected = (j_start + j + 1) * base_state
                found = state1[j]
                self.assertTrue(
                    (found == expected).all(),
                    'State is not properly created for index {} slice {}. '
                    'Expected: {}, Found: {}'.format(i, j, expected, found))

    def test_create_transition(self):
        """EpisodeRecorder correctly creates transitions from histories"""
        er = _make_episode_recorder(stack=13)

        n_actions = 3 * er.max_stack
        for i in range(n_actions):
            action, reward, terminal = i, 2*i, bool(i % 3)
            state = (i+1) * base_state
            er.put({
                'action': action, 'reward': reward,
                'state': state, 'terminal': terminal
            })
        for i in range(er.max_stack-1, n_actions):
            transition = er.get(i)
            j_start = i - er.max_stack + 1
            state0 = transition['state'][0]
            state1 = transition['state'][1]
            for j in range(er.max_stack):
                expected = (j_start + j) * base_state
                found = state0[j]
                self.assertTrue(
                    (expected == found).all(),
                    'Transition was not proeperly created. '
                    'state0 for index {}, slice {} is not correct. '
                    'Expected: {}, Found: {}'.format(i, j, expected, found))
                expected = (j_start + j + 1) * base_state
                found = state1[j]
                self.assertTrue(
                    (expected == found).all(),
                    'Transition was not proeperly created. '
                    'state1 for index {}, slice {} is not correct. '
                    'Expected: {}, Found: {}'.format(i, j, expected, found))
            expected = i
            found = transition['action']
            self.assertEqual(
                expected, found,
                'Transition was not proeperly created. '
                'action for index {} is not correct. '
                'Expected: {}, Found: {}'.format(i, expected, found))
            expected = 2 * i
            found = transition['reward']
            self.assertEqual(
                expected, found,
                'Transition was not proeperly created. '
                'reward for index {} is not correct. '
                'Expected: {}, Found: {}'.format(i, expected, found))
            expected = bool(i % 3)
            found = transition['terminal']
            self.assertEqual(
                expected, found,
                'Transition was not proeperly created. '
                'terminal for index {} is not correct. '
                'Expected: {}, Found: {}'.format(i, expected, found))

    def test_sample(self):
        """EpisodeRecorder correctly samples a transition"""
        er = _make_episode_recorder(stack=5)

        n_actions = 3 * er.max_stack
        for i in range(1, n_actions+1):
            action, reward, terminal = i, 2*i, bool(i % 3)
            state = (i+1) * base_state
            er.put({
                'action': action, 'reward': reward,
                'state': state, 'terminal': terminal,
            })

        for _ in range(1000):
            transition = er.sample()
            state0 = transition['state'][0]
            action = transition['action']
            reward = transition['reward']
            state1 = transition['state'][1]
            terminal = transition['terminal']
            self.assertTrue((action == state0[-1]).all())
            self.assertEqual(2 * action, reward)
            self.assertEqual(bool(action % 3), terminal)
            self.assertEqual(state0[1:], state1[:-1])


class TestTransitionRecorder(unittest.TestCase):
    def test_reset(self):
        """`reset` should create new EpisodeRecorder with empty record"""
        n_episodes = 100
        timesteps = 100

        tr = _create_transition_recorder(memory_size=n_episodes * timesteps)
        for _ in range(n_episodes):
            tr.reset(initial_data={'state': 0*base_state})
            expected = 0
            found = tr._recorder.n_data
            self.assertEqual(
                expected, found,
                '`recorder` is not properly reset. '
                '#actions expected is {}, instead of {}.'
                .format(expected, found))

    def test_truncate_recorders(self):
        """`truncate` should put current recorder to recorder archives"""
        n_episodes = 100
        timesteps = 10

        tr = _create_transition_recorder(memory_size=n_episodes*timesteps+1)
        self.assertEqual(0, len(tr._recorders))
        for i in range(n_episodes):
            tr.reset(initial_data={'state': 0*base_state})
            for j in range(timesteps):
                tr.record({
                    'state': j * base_state,
                    'action': j,
                    'reward': 2 * j,
                    'terminal': bool(j % 3)
                })
            tr.truncate()

            expected = i + 1
            found = len(tr._recorders)
            self.assertEqual(
                expected, found,
                '`recorder` was not properly put into `recorders`. '
                '#recorders expected is {}, instead of {}.'
                .format(expected, found))

    def test_truncate(self):
        """`truncate` retains at least one recorder or maximum records"""
        memory_size = 100
        tr = _create_transition_recorder(memory_size=memory_size)

        # Let's say we went through 10 episodes and we observed 10 actions in
        # each episode
        for i in range(10):
            tr.reset(initial_data={'state': 0*base_state})
            for j in range(10):
                tr.record({
                    'state': j * base_state,
                    'action': j,
                    'reward': 2 * j,
                    'terminal': bool(j % 3)
                })
            tr.truncate()

            # At the end of each episode, #actions we observed through all the
            # episodes is smaller than memory_size, so no recorder
            # should be removed.
            expected = i + 1
            found = len(tr._recorders)
            self.assertEqual(
                expected, found,
                '`truncate` method should not alter recorders when '
                'there are observations less than `n_records_to_retain`.'
                'Expected: {}, Found: {}'
                .format(expected, found))

        # Let's go through another episode with 50 actions.
        # After `truncate`, records from old episode should be removed.
        tr.reset(initial_data={'state': 0*base_state})
        for j in range(50):
            tr.record({
                'state': j * base_state,
                'action': j,
                'reward': 2 * j,
                'terminal': bool(j % 3)
            })
        tr.truncate()

        expected = 6  # The last six episodes: 10 + 10 + 10 + 10 + 10 + 50
        found = len(tr._recorders)
        self.assertEqual(
            expected, found,
            '`truncate` method should retain the latest recorders when '
            'there are observations more than `n_records_to_retain`.'
            'Expected: {}, Found: {}'
            .format(expected, found))

        # Let's go through another episode with actions
        # more than n_records_to_retain.
        # After `truncate`, only records from the last espisode should
        # be left.
        n_actions_to_add = memory_size + 1
        tr.reset(initial_data={'state': 0*base_state})
        for _ in range(n_actions_to_add):
            tr.record({
                'state': j * base_state,
                'action': j,
                'reward': 2 * j,
                'terminal': bool(j % 3)
            })
        tr.truncate()

        expected = 1
        found = len(tr._recorders)
        self.assertEqual(
            expected, found,
            '`truncate` method should retain the latest recorder when '
            'that recorder contains observations more than '
            '`n_records_to_retain`. Expected: {}, Found: {}'
            .format(expected, found))

        expected = n_actions_to_add
        found = tr.n_records
        self.assertEqual(
            expected, found,
            '`truncate` method should retain the latest recorder when '
            'that recorder contains observations more than '
            '`n_records_to_retain`. Expected: {}, Found: {}'
            .format(expected, found))

    def test_sampling_raises(self):
        """Sampling raises ValueError when there is no record"""
        n_samples = 32
        tr = _create_transition_recorder(memory_size=10)
        try:
            tr.sample(n_samples=n_samples)
        except ValueError:
            pass
        except Exception as error:  # pylint: disable=broad-except
            self.fail('Expected `ValueError` but got {}'.format(error))
        else:
            self.fail('Expected `ValueError` but no error was raised.')

    def test_sampling(self):
        n_samples = 32
        tr = _create_transition_recorder(memory_size=3*n_samples)

        for _ in range(10):
            i = np.random.randint(255)
            tr.reset(initial_data={'state': i*base_state})
            for _ in range(n_samples):
                j = np.random.randint(255)
                tr.record({
                    'action': i,
                    'reward': i,
                    'state': j * base_state,
                    'terminal': bool(i % 3),
                })
                i = j
            tr.truncate()

        samples = tr.sample(n_samples=n_samples)
        data = [
            samples['state'][0],
            samples['action'],
            samples['reward'],
            samples['state'][1],
            samples['terminal'],
        ]
        for state0, action, reward, state1, terminal in zip(*data):
            self.assertEqual(
                action, reward,
                'Incorrect action-reward pair is found.')
            self.assertEqual(
                bool(action % 3), terminal,
                'Incorrect action-terminal pair is found.')
            self.assertTrue(
                (action == state0[-1, ...]).all(),
                'Incorrect action-observation pair is found.')
            self.assertTrue(
                (state0[1:, ...] == state1[:-1, ...]).all(),
                'Incorrect pre_state-post_state pair is found.')

    def test_get_current_state(self):
        stack, transition = 4, 2
        tr = _create_transition_recorder(
            memory_size=200, batch_size=32, stack=stack)

        tr.reset(initial_data={'state': 0*base_state})
        n_actions = stack + transition
        for i in range(n_actions):
            tr.record({
                'action': i,
                'reward': i,
                'state': (i+1) * base_state,
                'terminal': bool(i % 3),
            })

        stacks = tr.get_last_stack()
        for i in range(stack):
            expected = n_actions - stack + i + 1
            found = stacks['state'][i]
            self.assertTrue(
                (expected == found).all(),
                'The last observations were not correctly converted to state. '
                'Value expected: {}, Found: {}.'.format(expected, found)
            )

        tr.reset(initial_data={'state': 0*base_state})
        n_actions2 = n_actions * 2
        for i in range(n_actions, n_actions2):
            tr.record({
                'action': i,
                'reward': i,
                'state': (i+1) * base_state,
                'terminal': bool(i % 3),
            })

        stacks = tr.get_last_stack()
        for i in range(stack):
            expected = n_actions2 - stack + i + 1
            found = stacks['state'][i]
            self.assertTrue(
                (expected == found).all(),
                'The last observations were not converted to state. '
                'Value expected: {}, Found: {}.'.format(expected, found)
            )

    def test_sampling_time(self):
        """Sampling should finish in reasonable time."""
        n_samples, stack = 32, 4
        tr = _create_transition_recorder(
            memory_size=1000000, batch_size=n_samples, stack=stack)

        for i in range(10000):
            tr.reset(initial_data={'state': 0*base_state})
            for _ in range(100):
                tr.record({
                    'action': i,
                    'reward': i,
                    'state': (i+1) * base_state,
                    'terminal': bool(i % 3),
                })
            tr.truncate()

        n_repeats = 1000
        t0 = time.time()
        for _ in range(n_repeats):
            tr.sample(n_samples)
        dt = (time.time() - t0) / n_repeats
        print('{} [sec]'.format(dt))

        threshold = 0.0013 if 'CIRCLECI' in os.environ else 0.0008
        self.assertTrue(
            dt < threshold,
            'Sampling is taking too much time ({} sec). '
            'Review implementation.'.format(dt)
        )
