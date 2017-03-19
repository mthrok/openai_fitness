"""Atari Environment based on Arcade Learning Environment"""
from __future__ import absolute_import

import sys
import os
import logging

import numpy as np
from scipy.misc import imresize
from ale_python_interface import ALEInterface

from luchador.util import StoreMixin, pprint_dict
from ..base import BaseEnvironment, Outcome

__all__ = ['ALEEnvironment']
_LG = logging.getLogger(__name__)
_DIR = os.path.dirname(os.path.abspath(__file__))
_ROM_DIR = os.path.join(_DIR, 'rom')


class ALEEnvironment(StoreMixin, BaseEnvironment):
    """Atari Environment"""
    @staticmethod
    def get_roms():
        """Get the list of ROMs available

        Returns:
          list of srting: Names of available ROMs
        """
        return [rom for rom in os.listdir(_ROM_DIR)
                if rom.endswith('.bin')]

    def _validate_args(
            self, mode, preprocess_mode,
            repeat_action, random_start, rom, **_):
        if mode not in ['test', 'train']:
            raise ValueError('`mode` must be either `test` or `train`')

        if preprocess_mode not in ['max', 'average']:
            raise ValueError(
                '`preprocess_mode` must be either `max` or `average`')

        if repeat_action < 1:
            raise ValueError(
                '`repeat_action` must be integer greater than 0')

        if random_start and random_start < 1:
            raise ValueError(
                '`random_start` must be `None` or integer greater than 0'
            )

        rom_path = os.path.join(_ROM_DIR, rom)
        if not os.path.isfile(rom_path):
            raise ValueError('ROM ({}) not found.'.format(rom))

    def __init__(
            self, rom,
            mode='train',
            width=None,
            height=None,
            grayscale=True,
            repeat_action=4,
            buffer_frames=2,
            preprocess_mode='max',
            minimal_action_set=True,
            random_seed=0,
            random_start=None,
            display_screen=False,
            play_sound=False,
            record_screen_path=None,
            record_sound_filename=None,
    ):
        """Initialize ALE Environment with the given parmeters

        Parameters
        ----------
        rom : str
            ROM name. Use `get_roms` for the list of available ROMs.

        mode : str
            When `train`, a loss of life is considered as terminal condition.
            When `test`, a loss of life is not considered as terminal
            condition.

        width, height : int or None
            Output screen size. If None the original size is used.

        grayscale : bool
            If True, output screen is gray scale and has no color channel.
            i.e. output shape == (h, w). Otherwise output screen has color
            channel with shape (h, w, 3)

        repeat_action : int
            When calling `step` method, action is repeated for this numebr of
            times internally, unless a terminal condition is met.

        minimal_action_set : bool
            When True, `n_actions` property reports actions only meaningfull to
            the loaded ROM. Otherwise all the 18 actions are dounted.

        random_seed : int
            ALE's random seed

        random_start : int or None
            When given,  at the beginning of each episode at most this number
            of frames are played with action == 0. This technique is used to
            acquire more diverse states of environment.

        buffer_frames : int
            The number of latest frame to preprocess.

        preprocess_mode : str
            Either `max` or `average`. When obtaining observation, pixel-wise
            maximum or average over buffered frames are taken before resizing

        display_screen : bool
            Display sceen when True.

        play_sound : bool
            Play sound

        record_screen_path : str
            Passed to ALE. Save the original screens into the path.

            Note
                that this is different from the observation returned by
                `step` method.

        record_screen_filename : str
            Passed to ALE. Save sound to a file.
        """
        if not rom.endswith('.bin'):
            rom += '.bin'

        self._store_args(
            rom=rom, mode=mode, width=width, height=height,
            grayscale=grayscale, repeat_action=repeat_action,
            buffer_frames=buffer_frames, preprocess_mode=preprocess_mode,
            minimal_action_set=minimal_action_set, random_seed=random_seed,
            random_start=random_start, display_screen=display_screen,
            play_sound=play_sound, record_screen_path=record_screen_path,
            record_sound_filename=record_sound_filename)

        if display_screen and sys.platform == 'darwin':
            import pygame
            pygame.init()

        self._buffer_index = None
        self.life_lost = False

        self._init_ale()
        self._init_buffer()
        self._init_resize()

    def _init_ale(self):
        ale = ALEInterface()
        ale.setBool('sound', self.args['play_sound'])
        ale.setBool('display_screen', self.args['display_screen'])
        ale.setInt('random_seed', self.args['random_seed'])

        # Frame skip is implemented separately
        ale.setInt('frame_skip', 1)
        ale.setBool('color_averaging', False)
        ale.setFloat('repeat_action_probability', 0.0)
        # Somehow this repeat_action_probability has unexpected effect on game.
        # The larger this value is, the more frames games take to restart.
        # And when 1.0 games completely hang
        # We are setting the default value of 0.0 here, expecting that
        # it has no effect as frame_skip == 1
        # This action repeating is agent's concern
        # so we do not implement an equivalent in our wrapper.

        if self.args['record_screen_path']:
            _LG.info('Recording screens: %s', self.args['record_screen_path'])
            if not os.path.exists(self.args['record_screen_path']):
                os.makedirs(self.args['record_screen_path'])
            ale.setString('record_screen_dir', self.args['record_screen_path'])

        if self.args['record_sound_filename']:
            _LG.info('Recording sound: %s', self.args['record_sound_filename'])
            record_sound_dir = os.path.dirname(
                self.args['record_sound_filename'])
            if not os.path.exists(record_sound_dir):
                os.makedirs(record_sound_dir)
            ale.setBool('sound', True)
            ale.setString(
                'record_sound_filename', self.args['record_sound_filename'])

        ale.loadROM(os.path.join(_ROM_DIR, self.args['rom']))

        self._ale = ale
        self._actions = (
            ale.getMinimalActionSet() if self.args['minimal_action_set'] else
            ale.getLegalActionSet()
        )

    def _init_buffer(self):
        orig_width, orig_height = self._ale.getScreenDims()
        channel = 1 if self.args['grayscale'] else 3
        n_frames = self.args['buffer_frames']

        buffer_shape = (
            (n_frames, orig_height, orig_width) if self.args['grayscale'] else
            (n_frames, orig_height, orig_width, channel)
        )
        self._frame_buffer = np.zeros(buffer_shape, dtype=np.uint8)
        self._buffer_index = 0

        self._get_raw_screen = (
            self._ale.getScreenGrayscale if self.args['grayscale'] else
            self._ale.getScreenRGB
        )

        self._get_screen = (
            self._get_max_buffer if self.args['preprocess_mode'] == 'max' else
            self._get_average_buffer
        )

    def _get_max_buffer(self):
        return np.max(self._frame_buffer, axis=0)

    def _get_average_buffer(self):
        return np.mean(self._frame_buffer, axis=0)

    def _init_resize(self):
        orig_width, orig_height = self._ale.getScreenDims()
        h = self.args['height'] or orig_height
        w = self.args['width'] or orig_width
        if h == orig_height and w == orig_width:
            self.resize = None
        else:
            self.resize = (h, w) if self.args['grayscale'] else (h, w, 3)

    ###########################################################################
    def __repr__(self):
        return str(self.args)

    def __str__(self):
        return pprint_dict(self.args)

    ###########################################################################
    @property
    def n_actions(self):
        return len(self._actions)

    ###########################################################################
    def _get_info(self):
        return {
            'lives': self._ale.lives(),
            'total_frame_number': self._ale.getFrameNumber(),
            'episode_frame_number': self._ale.getEpisodeFrameNumber(),
        }

    def _random_play(self):
        rand = self.args['random_start']
        repeat = 1 + (np.random.randint(rand) if rand else 0)
        return sum(self._step(0) for _ in range(repeat))

    def reset(self):
        """Reset game

        In test mode, the game is simply initialized. In train mode, if the
        game is in terminal state due to a life loss but not yet game over,
        then only life loss flag is reset so that the next game starts from
        the current state. Otherwise, the game is simply initialized.
        """
        reward = 0
        if (
                self.args['mode'] == 'test' or
                not self.life_lost or  # `reset` called in a middle of episode
                self._ale.game_over()  # all lives are lost
        ):
            self._ale.reset_game()
            reward += self._random_play()

        self.life_lost = False
        return Outcome(
            reward=reward,
            state=self._get_state(),
            terminal=self._is_terminal(),
            info=self._get_info(),
        )

    ###########################################################################
    # methods for `step` function
    def step(self, action):
        reward = 0
        action = self._actions[action]

        self.life_lost = False
        initial_lives = self._ale.lives()
        for _ in range(self.args['repeat_action']):
            reward += self._step(action)

            if not self._ale.lives() == initial_lives:
                self.life_lost = True

            terminal = self._is_terminal()
            if terminal:
                break

        return Outcome(
            reward=reward,
            state=self._get_state(),
            terminal=terminal,
            info=self._get_info(),
        )

    def _step(self, action):
        reward = self._ale.act(action)
        buffer_ = self._frame_buffer[self._buffer_index]
        self._get_raw_screen(screen_data=buffer_)
        self._buffer_index = (
            (self._buffer_index + 1) % self.args['buffer_frames'])
        return reward

    def _get_state(self):
        screen = self._get_screen()
        if self.resize:
            return imresize(screen, self.resize)
        return screen

    def _is_terminal(self):
        if self.args['mode'] == 'train':
            return self._ale.game_over() or self.life_lost
        return self._ale.game_over()
