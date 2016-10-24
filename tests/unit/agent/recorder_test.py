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


class TestEpisodeRecorder(unittest.TestCase):
    def test_record(self):
        """EpisodeRecorder correctly records histories"""
        ones = create_image(height=19, width=37, channel=None)
        er = EpisodeRecorder(initial_observation=(0*ones))

        observation_unit = 11
        n_actions = 5 * observation_unit
        for i in range(1, n_actions + 1):
            observation = i * ones
            action, reward, terminal = i, 2 * i, bool(i % 3)
            er.record(action, observation, reward, terminal)

        for i in range(n_actions + 1):
            expected = i * ones
            found = er.observations[i]
            self.assertTrue(
                (found == expected).all(),
                'Observation is not correctly recorded. '
                'Expected: {}, Found: {}.'.format(expected, found))

        for i in range(1, n_actions + 1):
            expected = i
            found = er.actions[i]
            self.assertEqual(
                expected, found,
                'Action is not correctly recorded. '
                'Expected: {}, Found: {}.'.format(expected, found))
            expected = 2 * i
            found = er.rewards[i]
            self.assertEqual(
                expected, found,
                'Reward is not correctly recorded. '
                'Expected: {}, Found: {}.'.format(expected, found))
            expected = bool(i % 3)
            found = er.terminals[i]
            self.assertEqual(
                expected, found,
                'Terminal is not correctly recorded. '
                'Expected: {}, Found: {}.'.format(expected, found))

    def test_create_state(self):
        """EpisodeRecorder correctly creates state from observations"""
        ones = create_image(height=19, width=37, channel=None)
        er = EpisodeRecorder(initial_observation=0 * ones)

        state_length = 5
        n_actions = 3 * state_length
        for i in range(1, n_actions+1):
            observation = i * ones
            action, reward, terminal = i, 2 * i, bool(i % 3)
            er.record(action, observation, reward, terminal)

        for i in range(state_length-1, n_actions):
            state = er._create_state(i, state_length)
            j_start = i - state_length + 1
            for j in range(state_length):
                expected = (j_start + j) * ones
                found = state[j]
                self.assertTrue(
                    (found == expected).all(),
                    'State is not properly created for index {} slice {}. '
                    'Expected: {}, Found: {}'.format(i, j, expected, found))

    def test_create_transition(self):
        """EpisodeRecorder correctly creates transitions from histories"""
        ones = create_image(height=19, width=37, channel=None)
        er = EpisodeRecorder(initial_observation=0*ones)

        state_length = 13
        n_actions = 3 * state_length
        for i in range(1, n_actions + 1):
            observation = i * ones
            action, reward, terminal = i, 2*i, bool(i % 3)
            er.record(action, observation, reward, terminal)

        for i in range(state_length, n_actions+1):
            transition = er._create_transition(i, state_length)
            j_start = i - state_length + 1
            for j in range(state_length):
                expected = (j_start + j - 1) * ones
                found = transition['pre_state'][j]
                self.assertTrue(
                    (expected == found).all(),
                    'Transition was not proeperly created. '
                    'state0 for index {}, slice {} is not correct. '
                    'Expected: {}, Found: {}'.format(i, j, expected, found))
                expected = (j_start + j) * ones
                found = transition['post_state'][j]
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
        ones = create_image(height=19, width=37, channel=None)
        er = EpisodeRecorder(initial_observation=0 * ones)

        state_length = 5
        n_actions = 3 * state_length
        for i in range(1, n_actions+1):
            er.record(i, i * ones, i, bool(i))

        for _ in range(1000):
            transition = er.sample(state_length)
            state0 = transition['pre_state']
            action = transition['action']
            reward = transition['reward']
            state1 = transition['post_state']
            terminal = transition['terminal']
            self.assertTrue((action == state0[-1] + 1).all())
            self.assertEqual(action, reward)
            self.assertEqual(bool(action), terminal)
            self.assertTrue((action == state1[-1]).all())


class TestTransitionRecorder(unittest.TestCase):
    def test_reset(self):
        """`reset` should create new EpisodeRecorder with empty record"""
        obs = create_image(height=19, width=37, channel=None)
        n_episodes = 100
        timesteps = 100

        tr = TransitionRecorder(
            state_length=4,
            memory_size=n_episodes * timesteps)

        for _ in range(n_episodes):
            tr.reset(initial_observation=obs)
            expected = 0
            found = len(tr.recorder)
            self.assertEqual(
                expected, found,
                '`recorder` is not properly reset. '
                '#actions expected is {}, instead of {}.'
                .format(expected, found))

    def test_truncate_recorders(self):
        """`truncate` should put current recorder to recorder archives"""
        obs = create_image(height=19, width=37, channel=None)
        n_episodes = 100
        timesteps = 100

        tr = TransitionRecorder(
            state_length=4,
            memory_size=n_episodes * timesteps)

        for i in range(n_episodes):
            tr.reset(initial_observation=obs)
            expected = i
            found = len(tr.recorders)
            self.assertEqual(
                expected, found,
                '`recorder` was not properly put into `recorders`. '
                '#recorders expected is {}, instead of {}.'
                .format(expected, found))

            for _ in range(timesteps):
                tr.record(1, obs, 1, True)
            tr.truncate()

    def test_truncate(self):
        """`truncate` retains at least one recorder or maximum records"""
        obs = create_image(height=19, width=37, channel=None)
        memory_size = 100
        tr = TransitionRecorder(
            memory_size=memory_size,
            state_length=4)

        # Let's say we went through 10 episodes and we observed 10 actions in
        # each episode
        for i in range(10):
            tr.reset(initial_observation=obs)
            for _ in range(10):
                tr.record(1, obs, 1, True)
            tr.truncate()

            # At the end of each episode, #actions we observed through all the
            # episodes is smaller than memory_size, so no recorder
            # should be removed.
            expected = i + 1
            found = len(tr.recorders)
            self.assertEqual(
                expected, found,
                '`truncate` method should not alter recorders when '
                'there are observations less than `n_records_to_retain`.'
                'Expected: {}, Found: {}'
                .format(expected, found))

        # Let's go through another episode with 50 actions.
        # After `truncate`, records from old episode should be removed.
        tr.reset(initial_observation=obs)
        for _ in range(50):
            tr.record(1, obs, 1, True)
        tr.truncate()

        expected = 6  # The last six episodes: 10 + 10 + 10 + 10 + 10 + 50
        found = len(tr.recorders)
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
        tr.reset(initial_observation=obs)
        for _ in range(n_actions_to_add):
            tr.record(1, obs, 1, True)
        tr.truncate()

        expected = 1
        found = len(tr.recorders)
        self.assertEqual(
            expected, found,
            '`truncate` method should retain the latest recorder when '
            'that recorder contains observations more than '
            '`n_records_to_retain`. Expected: {}, Found: {}'
            .format(expected, found))

        expected = n_actions_to_add
        found = len(tr)
        self.assertEqual(
            expected, found,
            '`truncate` method should retain the latest recorder when '
            'that recorder contains observations more than '
            '`n_records_to_retain`. Expected: {}, Found: {}'
            .format(expected, found))

    def test_sampling_raises(self):
        """Sampling raises ValueError when there is no record"""
        n_samples = 33
        tr = TransitionRecorder()
        try:
            tr.sample(n_samples=n_samples)
        except ValueError:
            pass
        except Exception as e:
            self.fail('Expected `ValueError` but got {}'.format(e))
        else:
            self.fail('Expected `ValueError` but no error was raised.')

    def _test_sampling(self, fmt):
        height, width = 19, 37
        n_samples, state_length = 33, 4
        ones = create_image(height=height, width=width, channel=None)
        tr = TransitionRecorder(
            memory_size=3*n_samples,
            state_length=state_length,
            state_width=width, state_height=height,
            batch_size=n_samples, data_format=fmt)

        for _ in range(10):
            tr.reset(ones)
            for _ in range(n_samples):
                i = np.random.randint(255)
                tr.record(i, i*ones, i, bool(i % 3))
            tr.truncate()

        samples = tr.sample(n_samples=n_samples)
        keys = ['pre_states', 'actions', 'rewards', 'post_states', 'terminals']
        data = [samples[k] for k in keys]
        for pre, action, reward, post, terminal in zip(*data):
            last_obs = post[..., -1] if fmt == 'NHWC' else post[-1, ...]
            common_pre = pre[..., 1:] if fmt == 'NHWC' else pre[1:, ...]
            common_post = post[..., :-1] if fmt == 'NHWC' else post[:-1, ...]
            self.assertEqual(action, reward,
                             'Incorrect action-reward pair is found.')
            self.assertEqual(bool(action % 3), terminal,
                             'Incorrect action-terminal pair is found.')
            self.assertTrue((action == last_obs).all(),
                            'Incorrect action-observation pair is found.')
            self.assertTrue((common_pre == common_post).all(),
                            'Incorrect pre_state-post_state pair is found.')

    def test_sampling_nchw(self):
        """Samples the correct set of transitions when format is "NCHW" """
        self._test_sampling('NCHW')

    def test_sampling_nhwc(self):
        """Samples the correct set of transitions when format is "NHWC" """
        self._test_sampling('NHWC')

    def _test_get_current_state(self, fmt):
        height, width = 19, 37
        state_length = 4
        ones = create_image(height=height, width=width, channel=None)
        tr = TransitionRecorder(
            memory_size=200,
            state_length=state_length,
            state_width=width, state_height=height,
            batch_size=31, data_format=fmt)

        tr.reset(0*ones)
        n_actions = 2 * state_length
        for i in range(1, n_actions + 1):
            observation = i * ones
            tr.record(0, observation, 0, True)

        state = tr.get_current_state()
        for i in range(state_length):
            expected = n_actions + 1 - state_length + i
            found = state[:, i, ...] if fmt == 'NCHW' else state[..., i]
            self.assertTrue(
                (expected == found).all(),
                'The last observations were not correctly converted to state. '
                'Value expected: {}, Found: {}.'.format(expected, found)
            )

        tr.reset(0 * ones)
        n_actions2 = n_actions * 2
        for i in range(n_actions + 1, n_actions2):
            observation = i * ones
            tr.record(0, observation, 0, True)

        state = tr.get_current_state()
        for i in range(state_length):
            expected = n_actions2 - state_length + i
            found = state[0, i, ...] if fmt == 'NCHW' else state[0, ..., i]
            self.assertTrue(
                (expected == found).all(),
                'The last observations were not converted to state. '
                'Value expected: {}, Found: {}.'.format(expected, found)
            )

    def test_get_current_state_nchw(self):
        """Last observations are retrieved in "NCHW" format"""
        self._test_get_current_state('NCHW')

    def test_get_current_state_nhwc(self):
        """Last observations are retrieved in "NHWC" format"""
        self._test_get_current_state('NHWC')

    @unittest.skipIf('CIRCLECI' in os.environ, 'Skipping sampling time')
    def test_sampling_time(self):
        """Sampling should finish in reasonable time."""
        height, width = 1, 1
        n_samples, state_length = 33, 4
        ones = create_image(height=height, width=width, channel=None)
        tr = TransitionRecorder(
            memory_size=1000000,
            state_length=state_length,
            state_width=width, state_height=height,
            batch_size=n_samples)

        for i in range(10000):
            tr.reset(ones)
            for _ in range(100):
                tr.record(i, ones, i, True)
            tr.truncate()

        n_repeats = 1000
        t0 = time.time()
        for _ in range(n_repeats):
            tr.sample(n_samples)
        dt = (time.time() - t0) / n_repeats
        print('{} [sec]'.format(dt))
        self.assertTrue(
            dt < 0.0008,
            'Sampling is taking too much time ({} sec). '
            'Review implementation.'.format(dt)
        )
