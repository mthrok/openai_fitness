from __future__ import absolute_import

import unittest

from luchador.env.ale import ALEEnvironment as ALE


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
            width=None,
            height=None,
            grayscale=True,
            repeat_action=1,
            random_start=None,
        )

        outcome = ale.reset()
        lives_before = outcome.state['lives']
        while True:
            outcome = ale.step(1)
            if ale._ale.game_over():
                break
            lives_after = outcome.state['lives']
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
            width=None,
            height=None,
            grayscale=True,
            repeat_action=1,
            random_start=None,
        )

        outcome = ale.reset()
        lives_before = outcome.state['lives']
        while True:
            outcome = ale.step(1)
            if ale._ale.game_over():
                break

            lives_after = outcome.state['lives']
            if lives_before == lives_after:
                continue
            lives_before = lives_after

            self.assertTrue(
                outcome.terminal,
                'A life loss must be considered as terminal in `train` mode'
            )

    def test_no_resize_grayscale(self):
        """Observation width and height equal to the original screen size"""
        ale = ALE(
            rom='breakout',
            width=None,
            height=None,
            grayscale=True,
        )

        ale.reset()
        outcome = ale.step(1)
        expected = (210, 160)
        found = outcome.observation.shape
        self.assertEqual(
            expected, found,
            'Observation shape must equal to the original screen size, '
            'when `width` and `height` are `None`.'
        )

    def test_no_resize_rgb(self):
        """Observation width and height equal to the original screen size"""
        ale = ALE(
            rom='breakout',
            width=None,
            height=None,
            grayscale=False,
        )

        ale.reset()
        outcome = ale.step(1)
        expected = (210, 160)
        found = outcome.observation.shape[:2]
        self.assertEqual(
            expected, found,
            'Observation shape must equal to the original screen size, '
            'when `width` and `height` are `None`.'
        )

    def test_resize_width_grayscale(self):
        """Observation size equals to the given size"""
        width = 84

        ale = ALE(
            rom='breakout',
            width=width,
            height=None,
            grayscale=True,
        )

        ale.reset()
        outcome = ale.step(1)
        expected = (210, width)
        found = outcome.observation.shape
        self.assertEqual(
            expected, found,
            'Observation must be resized when width is given.'
        )

    def test_resize_height_grayscale(self):
        """Observation size equals to the given size"""
        height = 84

        ale = ALE(
            rom='breakout',
            width=None,
            height=height,
            grayscale=True,
        )

        ale.reset()
        outcome = ale.step(1)
        expected = (height, 160)
        found = outcome.observation.shape
        self.assertEqual(
            expected, found,
            'Observation must be resized when height is given.'
        )

    def test_resize_width_and_height_grayscale(self):
        """Observation size equals to the given size"""
        width, height = 84, 84

        ale = ALE(
            rom='breakout',
            width=width,
            height=height,
            grayscale=True,
        )

        ale.reset()
        outcome = ale.step(1)
        expected = (height, width)
        found = outcome.observation.shape
        self.assertEqual(
            expected, found,
            'Observation must be resized when both width and height are given.'
        )

    def test_resize_width_rgb(self):
        """Observation size equals to the given size"""
        width = 84

        ale = ALE(
            rom='breakout',
            width=width,
            height=None,
            grayscale=False,
        )

        ale.reset()
        outcome = ale.step(1)
        expected = (210, width)
        found = outcome.observation.shape[:2]
        self.assertEqual(
            expected, found,
            'Observation must be resized when width is given.'
        )

    def test_resize_height_rgb(self):
        """Observation size equals to the given size"""
        height = 84

        ale = ALE(
            rom='breakout',
            width=None,
            height=height,
            grayscale=False,
        )

        ale.reset()
        outcome = ale.step(1)
        expected = (height, 160)
        found = outcome.observation.shape[:2]
        self.assertEqual(
            expected, found,
            'Observation must be resized when height is given.'
        )

    def test_resize_width_and_height_rgb(self):
        """Observation size equals to the given size"""
        width, height = 84, 84

        ale = ALE(
            rom='breakout',
            width=width,
            height=height,
            grayscale=False,
        )

        ale.reset()
        outcome = ale.step(1)
        expected = (height, width)
        found = outcome.observation.shape[:2]
        self.assertEqual(
            expected, found,
            'Observation must be resized when both width and height is given.'
        )

    def test_rgb_observation_color_channel_without_resize(self):
        """Observation has color channel when grayscale=False"""
        ale = ALE(
            rom='breakout',
            grayscale=False,
        )

        ale.reset()
        outcome = ale.step(1)
        observation = outcome.observation
        self.assertTrue(len(observation.shape) == 3,
                        'Color channel is missing')
        self.assertTrue(observation.shape[2] == 3,
                        'Incorrect number of color channels')

    def test_rgb_observation_color_channel_with_resize(self):
        """Observation has color channel when grayscale=False"""
        ale = ALE(
            rom='breakout',
            grayscale=False,
            width=84, height=84,
        )

        ale.reset()
        outcome = ale.step(1)
        observation = outcome.observation
        self.assertTrue(len(observation.shape) == 3,
                        'Color channel is missing')
        self.assertTrue(observation.shape[2] == 3,
                        'Incorrect number of color channels')

    def test_grayscale_observation_color_channel_without_resize(self):
        """Observation has color channel when grayscale=True"""
        ale = ALE(
            rom='breakout',
            grayscale=True,
        )

        ale.reset()
        outcome = ale.step(1)
        observation = outcome.observation
        self.assertTrue(len(observation.shape) == 2)

    def test_grayscale_observation_color_channel_with_resize(self):
        """Observation has color channel when grayscale=True"""
        ale = ALE(
            rom='breakout',
            grayscale=True,
            width=84, height=84,
        )

        ale.reset()
        outcome = ale.step(1)
        observation = outcome.observation
        self.assertTrue(len(observation.shape) == 2)

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

                frame = outcome.state['episode_frame_number']
                self.assertEqual(frame - last_frame, repeat_action)
                last_frame = frame

    def test_buffer_frame_rgb(self):
        """_frame_buffer contains the last raw frames given as buffer_frames"""
        buffer_frames = 4
        ale = ALE(
            rom='breakout',
            mode='train',
            repeat_action=1,
            buffer_frames=buffer_frames,
            grayscale=False,
        )

        ale.reset()
        frame = ale._get_raw_screen()
        self.assertTrue((frame == ale._frame_buffer[0]).all())

        for i in range(1, buffer_frames):
            ale.step(1)
            frame = ale._get_raw_screen()
            self.assertTrue((frame == ale._frame_buffer[i]).all())

        for _ in range(10):
            for i in range(buffer_frames):
                ale.step(1)
                frame = ale._get_raw_screen()
                self.assertTrue((frame == ale._frame_buffer[i]).all())

    def test_buffer_frame_grayscale(self):
        """_frame_buffer contains the last raw frames given as buffer_frames"""
        buffer_frames = 4
        ale = ALE(
            rom='breakout',
            mode='train',
            repeat_action=1,
            buffer_frames=buffer_frames,
            grayscale=True,
        )

        ale.reset()
        frame = ale._get_raw_screen()
        for i in range(1, buffer_frames):
            ale.step(1)
            frame = ale._get_raw_screen()
            frame = frame[:, :, 0]
            self.assertTrue((frame == ale._frame_buffer[i]).all())

        for _ in range(10):
            for i in range(buffer_frames):
                ale.step(1)
                frame = ale._get_raw_screen()
                frame = frame[:, :, 0]
                self.assertTrue((frame == ale._frame_buffer[i]).all())

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
            frame = outcome.state['episode_frame_number']
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
            fr0 = outcome.state['episode_frame_number']
            outcome = ale.reset()
            fr1 = outcome.state['episode_frame_number']
            self.assertEqual(
                fr1, fr0,
                'New episode should not start at reset '
                'when mode==train and not game_over'
            )
        outcome = ale.reset()
        fr = outcome.state['episode_frame_number']
        self.assertEqual(
            fr, 1,
            'New episode should be started when `reset` is called on game_over'
        )
