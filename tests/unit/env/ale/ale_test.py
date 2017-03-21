from __future__ import absolute_import

import unittest

import numpy as np

from luchador.env.ale import ALEEnvironment as ALE


class ALEEnvShapeTest(unittest.TestCase):
    longMessage = True

    def _test(self, width=160, height=210, grayscale=True):
        ale = ALE(
            rom='breakout',
            width=width, height=height,
            grayscale=grayscale,
        )

        ale.reset()
        outcome = ale.step(1)
        channel = 1 if grayscale else 3
        self.assertEqual(outcome.state.shape, (channel, height, width))

    def test_no_resize(self):
        """State shape equals to the original screen size"""
        self._test(grayscale=True)

    def test_resize_width(self):
        """State shape equals to the given size"""
        self._test(width=84, grayscale=True)

    def test_resize_height(self):
        """State shape equals to the given size"""
        self._test(height=84, grayscale=True)

    def test_resize_width_height(self):
        """State shape equals to the given size"""
        self._test(height=84, width=84, grayscale=True)

    def test_no_resize_color(self):
        """State shape equals to the original screen size"""
        self._test(grayscale=False)

    def test_resize_width_color(self):
        """State shape equals to the given size"""
        self._test(width=84, grayscale=False)

    def test_resize_height_color(self):
        """State shape equals to the given size"""
        self._test(height=84, grayscale=False)

    def test_resize_width_height_color(self):
        """State shape equals to the given size"""
        self._test(height=84, width=84, grayscale=False)


def _test_buffer(grayscale):
    # pylint: disable=protected-access
    buffer_frames = 4
    ale = ALE(
        rom='breakout',
        mode='train',
        repeat_action=1,
        buffer_frames=buffer_frames,
        grayscale=grayscale,
    )
    buffer_ = ale._preprocessor._buffer

    ale.reset()
    frame = ale._get_raw_screen().transpose((2, 0, 1))
    np.testing.assert_equal(frame, buffer_[0])

    for i in range(1, buffer_frames):
        ale.step(1)
        frame = ale._get_raw_screen().transpose((2, 0, 1))
        np.testing.assert_equal(frame, buffer_[i])

    for _ in range(10):
        for i in range(buffer_frames):
            ale.step(1)
            frame = ale._get_raw_screen().transpose((2, 0, 1))
            np.testing.assert_equal(frame, buffer_[i])


class PreprocessorTest(unittest.TestCase):
    # pylint: disable=no-self-use
    def test_buffer_frame(self):
        """The latest frame is correctly passed to preprocessor buffer"""
        _test_buffer(grayscale=True)

    def test_buffer_frame_color(self):
        """The latest frame is correctly passed to preprocessor buffer"""
        _test_buffer(grayscale=False)


class ALEEnvironmentTest(unittest.TestCase):
    longMessage = True

    def test_rom_availability(self):
        """ROMs are available"""
        self.assertEqual(
            len(ALE.get_roms()), 61,
            'Not all the ALE ROMS are found. '
            'Run `python setup.py download_ale` from root directory '
            'to download ALE ROMs then re-install luchador.'
        )

    def test_test_mode_terminal_condition(self):
        """Loss of life is not considered terminal condition in `test` mode"""
        ale = ALE(
            rom='breakout',
            mode='test',
            grayscale=True,
            repeat_action=1,
            random_start=None,
        )

        outcome = ale.reset()
        lives_before = outcome.info['lives']
        while True:
            outcome = ale.step(1)
            if ale._ale.game_over():
                break
            lives_after = outcome.info['lives']
            if lives_before == lives_after:
                continue
            lives_before = lives_after

            self.assertFalse(
                outcome.terminal,
                'A life loss must not be considered as terminal in `test` mode'
            )

    def test_train_mode_terminal_condition(self):
        """Loss of life is not considered terminal condition in `test` mode"""
        ale = ALE(
            rom='breakout',
            mode='train',
            grayscale=True,
            repeat_action=1,
            random_start=None,
        )

        outcome = ale.reset()
        lives_before = outcome.info['lives']
        while True:
            outcome = ale.step(1)
            if ale._ale.game_over():
                break

            lives_after = outcome.info['lives']
            if lives_before == lives_after:
                continue
            lives_before = lives_after

            self.assertTrue(
                outcome.terminal,
                'A life loss must be considered as terminal in `train` mode'
            )

    def test_repeat_action(self):
        """`step` advances the number of frames given as `repeat_action`"""
        for repeat_action in [1, 4]:
            repeat_action = 1
            ale = ALE(
                rom='breakout',
                mode='test',
                repeat_action=repeat_action,
            )

            ale.reset()
            last_frame = 1
            while True:
                outcome = ale.step(1)

                if outcome.terminal:
                    break

                frame = outcome.info['episode_frame_number']
                self.assertEqual(frame - last_frame, repeat_action)
                last_frame = frame

    def test_random_start(self):
        """Episode starts from frame number in range of [1, `random_start`]"""
        random_start = 30
        ale = ALE(
            rom='breakout',
            random_start=random_start,
        )
        n_try = 100000
        checked = [False] * random_start
        for _ in range(n_try):
            outcome = ale.reset()
            frame = outcome.info['episode_frame_number']
            checked[frame - 1] = True

            if sum(checked) == random_start:
                break
        else:
            self.fail(
                'Not all starting frame (1 - {}) was observed after {} reset.'
                .format(random_start, n_try)
            )

    def test_train_reset(self):
        """reset does not reset game in train mode when game is not yet over"""
        ale = ALE(
            'breakout',
            mode='train',
            random_start=0,
        )
        ale.reset()
        while True:
            outcome = ale.step(action=1)
            if not outcome.terminal:
                continue
            if ale._ale.game_over():
                break
            fr0 = outcome.info['episode_frame_number']
            outcome = ale.reset()
            fr1 = outcome.info['episode_frame_number']
            self.assertEqual(
                fr1, fr0,
                'New episode should not start at reset '
                'when mode==train and not game_over'
            )
        outcome = ale.reset()
        fr = outcome.info['episode_frame_number']
        self.assertEqual(
            fr, 1,
            'New episode should be started when `reset` is called on game_over'
        )
