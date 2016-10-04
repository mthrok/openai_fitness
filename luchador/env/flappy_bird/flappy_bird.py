from __future__ import absolute_import

from itertools import cycle

import pygame
import numpy as np

from . import util
from ..base import BaseEnvironment, Outcome


# Constants
fps = 30
screen_width = 288
screen_height = 512

# Global object
FPSCLOCK, SCREEN = None, None


def _init_pygame():
    global SCREEN, FPSCLOCK
    pygame.init()
    FPSCLOCK = pygame.time.Clock()
    SCREEN = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption('Flappy Bird')


def _update_display():
    pygame.display.update()
    FPSCLOCK.tick(fps)


def _get_screen():
    pygame.surfarray.array3d(pygame.display.get_surface())


def get_index_generator(repeat=5, pattern=[0, 1, 2, 1]):
    indices = []
    for val in pattern:
        indices.extend([val] * repeat)
    return cycle(indices)


def pixel_collides(rect1, rect2, hitmask1, hitmask2):
    rect = rect1.clip(rect2)
    if rect.width == 0 or rect.height == 0:
        return False

    x1, y1 = rect.x - rect1.x, rect.y - rect1.y
    x2, y2 = rect.x - rect2.x, rect.y - rect2.y

    for x in xrange(rect.width):
        for y in xrange(rect.height):
            if hitmask1[x1+x][y1+y] and hitmask2[x2+x][y2+y]:
                return True
    return False


class FlappyBird(BaseEnvironment):
    def __init__(self, random_seed=None):
        _init_pygame()
        self._rng = np.random.RandomState(seed=random_seed)
        self._load_assets()

    def _load_assets(self):
        self._images, self._hitmasks = util.load_images()
        self._sounds = util.load_sounds()

    @property
    def n_actions(self):
        return 2

    ###########################################################################
    def reset(self):
        self.score = 0
        self._reset_background()
        self._reset_ground()
        self._reset_player()
        self._reset_pipe()
        self._draw()
        return Outcome(observation=_get_screen(), terminal=False, reward=0)

    def _get_new_pipe(self):
        """returns a randomly generated pipe"""
        x = screen_width + 10
        max_y = int(self.ground_y * 0.6 - self.pipe_gap_size) + 1
        base_y = self._rng.randint(0, max_y)
        base_y += int(self.ground_y * 0.2)

        upper_pipe = {'x': x, 'y': base_y - self.pipe_h}
        lower_pipe = {'x': x, 'y': base_y + self.pipe_gap_size}
        return upper_pipe, lower_pipe

    def _reset_background(self):
        index = self._rng.randint(1, len(self._images['backgrounds'])) - 1
        self._images['background'] = self._images['backgrounds'][index]

    def _reset_player(self):
        index = self._rng.randint(1, len(self._images['players'])) - 1
        self._images['player'] = self._images['players'][index]
        self._hitmasks['player'] = self._hitmasks['players'][index]

        self.player_g = 1   # players downward accleration
        self.player_v = -9
        self.player_v_max = 10   # max vel along Y, max descend speed
        self.player_v_flapped = -9   # players speed on flapping
        self.player_w = self._images['player'][0].get_width()
        self.player_h = self._images['player'][0].get_height()
        self.player_x = int(screen_width * 0.2)
        self.player_y = int((screen_height - self.player_h) / 2)

        self.motion_indices = get_index_generator(repeat=3)
        self.motion_index = self.motion_indices.next()

    def _reset_pipe(self):
        index = self._rng.randint(1, len(self._images['pipes'])) - 1
        self._images['pipe'] = self._images['pipes'][index]
        self._hitmasks['pipe'] = self._hitmasks['pipes'][index]

        self.pipe_w = self._images['pipe'][0].get_width()
        self.pipe_h = self._images['pipe'][0].get_height()
        self.pipe_v = -4
        self.pipe_gap_size = 100  # gap between upper and lower part of pipe

        self.upperPipes, self.lowerPipes = [], []
        for i in range(2):
            upper_pipe, lower_pipe = self._get_new_pipe()
            x = screen_width + 200 + (i * screen_width / 2)
            self.upperPipes.append({'x': x, 'y': upper_pipe['y']})
            self.lowerPipes.append({'x': x, 'y': lower_pipe['y']})

    def _reset_ground(self):
        ground_w = self._images['ground'].get_width()
        bg_w = self._images['background'].get_width()
        self.ground_x = 0
        self.ground_y = screen_height * 0.79
        self.ground_x_shift = ground_w - bg_w

    ###########################################################################
    def step(self, flapped):
        self.motion_index = self.motion_indices.next()
        self._move_player(flapped)
        self._move_ground()
        self._move_pipes()

        if self._crashed():
            self._sounds['hit'].play()
            reward = 0
            terminal = True
        else:
            reward = self._check_reward()
            terminal = False
        if reward:
            self._sounds['point'].play()
        self.score += reward

        self._draw()
        return Outcome(observation=_get_screen(),
                       terminal=terminal, reward=reward)

    def _move_pipes(self):
        # move pipes to left
        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            uPipe['x'] += self.pipe_v
            lPipe['x'] += self.pipe_v
        # add new pipe when first pipe is about to touch left of screen
        if 0 < self.upperPipes[0]['x'] < 5:
            upper_pipe, lower_pipe = self.get_new_pipe()
            self.upperPipes.append(upper_pipe)
            self.lowerPipes.append(lower_pipe)
        # remove first pipe if its out of the screen
        if self.upperPipes[0]['x'] < -self.pipe_w:
            self.upperPipes.pop(0)
            self.lowerPipes.pop(0)

    def _move_ground(self):
        self.ground_x = (self.ground_x - 100) % (-self.ground_x_shift)

    def _move_player(self, flapped):
        if flapped:
            if self.player_y >= 0:
                self.player_v = self.player_v_flapped
                self._sounds['wing'].play()
        elif self.player_v < self.player_v_max:
            self.player_v += self.player_g

        self.player_y += self.player_v
        self.player_y = min(self.player_y, self.ground_y - self.player_h)

    def _check_reward(self):
        player_x = self.player_x + self.player_w / 2
        for pipe in self.upperPipes:
            pipe_x = pipe['x'] + self.pipe_w / 2
            if pipe_x <= player_x < pipe_x + 4:
                return 1
        return 0

    def _crashed(self):
        if self.player_y + self.player_h + 1 >= self.ground_y:
            return True  # crashed into the ground

        u_mask, l_mask = self._hitmasks['pipe']
        p_mask = self._hitmasks['player'][self.motion_index]
        player = pygame.Rect(
            self.player_x, self.player_y, self.player_w, self.player_h)
        for u, l in zip(self.upperPipes, self.lowerPipes):
            u_pipe = pygame.Rect(u['x'], u['y'], self.pipe_w, self.pipe_h)
            l_pipe = pygame.Rect(l['x'], l['y'], self.pipe_w, self.pipe_h)
            if(
                    pixel_collides(player, u_pipe, p_mask, u_mask) or
                    pixel_collides(player, l_pipe, p_mask, l_mask)
            ):
                return True
        return False

    ###########################################################################
    def _draw_bg(self):
        SCREEN.blit(self._images['background'], (0, 0))

    def _draw_pipes(self):
        for u, l in zip(self.upperPipes, self.lowerPipes):
            SCREEN.blit(self._images['pipe'][0], (u['x'], u['y']))
            SCREEN.blit(self._images['pipe'][1], (l['x'], l['y']))

    def _draw_ground(self):
        SCREEN.blit(self._images['ground'], (self.ground_x, self.ground_y))

    def _draw_player(self):
        size = (self.player_x, self.player_y)
        SCREEN.blit(self._images['player'][self.motion_index], size)

    def _draw_digit(self, digit, x):
        SCREEN.blit(self._images['numbers'][digit], (x, screen_height * 0.1))

    def _draw_score(self):
        digits = [int(x) for x in list(str(self.score))]
        widths = [self._images['numbers'][digit].get_width()
                  for digit in digits]

        x = (screen_width - sum(widths)) / 2
        for digit, width in zip(digits, widths):
            self._draw_digit(digit, x)
            x += width

    def _draw(self):
        self._draw_bg()
        self._draw_pipes()
        self._draw_ground()
        self._draw_score()
        self._draw_player()
        _update_display()
