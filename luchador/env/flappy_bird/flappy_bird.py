from __future__ import absolute_import

import random
from itertools import cycle

import pygame

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


def pixelCollision(rect1, rect2, hitmask1, hitmask2):
    """Checks if two objects collide and not just their rects"""
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
    def __init__(self):
        _init_pygame()
        self._images, self._hitmasks = util.load_images()
        self._sounds = util.load_sounds()

        self.base_y = screen_height * 0.79

    @property
    def n_actions(self):
        return 2

    def _get_new_pipe(self):
        """returns a randomly generated pipe"""
        x = screen_width + 10
        base = random.randrange(0, int(self.base_y * 0.6 - self.pipe_gap_size))
        base += int(self.base_y * 0.2)

        upper_pipe = {'x': x, 'y': base - self.pipe_h}
        lower_pipe = {'x': x, 'y': base + self.pipe_gap_size}
        return upper_pipe, lower_pipe

    def _set_random_objects(self):
        index = random.randint(1, len(self._images['backgrounds'])) - 1
        self._images['background'] = self._images['backgrounds'][index]

        index = random.randint(1, len(self._images['players'])) - 1
        self._images['player'] = self._images['players'][index]
        self._hitmasks['player'] = self._hitmasks['players'][index]

        index = random.randint(1, len(self._images['pipes'])) - 1
        self._images['pipe'] = self._images['pipes'][index]
        self._hitmasks['pipe'] = self._hitmasks['pipes'][index]

    def reset(self):
        self._set_random_objects()

        self.score = 0
        self.player_a = -9   # players speed on flapping
        self.player_g = 1   # players downward accleration
        self.player_v = -9
        self.player_v_max = 10   # max vel along Y, max descend speed
        self.player_w = self._images['player'][0].get_width()
        self.player_h = self._images['player'][0].get_height()
        self.player_x = int(screen_width * 0.2)
        self.player_y = int((screen_height - self.player_h) / 2)

        self.pipe_w = self._images['pipe'][0].get_width()
        self.pipe_h = self._images['pipe'][0].get_height()
        self.pipe_v = -4
        self.pipe_gap_size = 100  # gap between upper and lower part of pipe

        self.motion_indices = get_index_generator(repeat=3)
        self.motion_index = self.motion_indices.next()

        base_w = self._images['base'].get_width()
        bg_w = self._images['background'].get_width()
        self.base_shift = base_w - bg_w
        self.base_x = 0

        self.upperPipes, self.lowerPipes = [], []
        for i in range(2):
            upper_pipe, lower_pipe = self._get_new_pipe()
            x = screen_width + 200 + (i * screen_width / 2)
            self.upperPipes.append({'x': x, 'y': upper_pipe['y']})
            self.lowerPipes.append({'x': x, 'y': lower_pipe['y']})

        self._draw()
        _update_display()
        return Outcome(observation=_get_screen(), terminal=False, reward=0)

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

    def _move_base(self):
        self.base_x = (self.base_x - 100) % (-self.base_shift)

    def _isCrashed(self):
        """returns True if player collders with base or pipes."""
        # if player crashes into ground
        if self.player_y + self.player_h >= self.base_y - 1:
            return True

        u_mask, l_mask = self._hitmasks['pipe']
        p_mask = self._hitmasks['player'][self.motion_index]

        player = pygame.Rect(self.player_x, self.player_y,
                             self.player_w, self.player_h)
        for u, l in zip(self.upperPipes, self.lowerPipes):
            # upper and lower pipe rects
            u_pipe = pygame.Rect(u['x'], u['y'], self.pipe_w, self.pipe_h)
            l_pipe = pygame.Rect(l['x'], l['y'], self.pipe_w, self.pipe_h)
            if(
                    pixelCollision(player, u_pipe, p_mask, u_mask) or
                    pixelCollision(player, l_pipe, p_mask, l_mask)
            ):
                return True
        return False

    def step(self, flapped):
        terminal = False
        reward = 0
        if flapped:
            if self.player_y > -2 * self.player_h:
                self.player_v = self.player_a
                self._sounds['wing'].play()
        elif self.player_v < self.player_v_max:
            self.player_v += self.player_g

        if self._isCrashed():
            terminal = True
            self._sounds['hit'].play()
        else:
            playerMidPos = self.player_x + self.player_w / 2
            for pipe in self.upperPipes:
                pipeMidPos = pipe['x'] + self.pipe_w / 2
                if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                    reward += 1
                    self._sounds['point'].play()
            self.score += reward

        # player_index base_x change
        self.motion_index = self.motion_indices.next()
        self._move_base()

        player_h = self._images['player'][self.motion_index].get_height()
        self.player_y += min(
            self.player_v, self.base_y - self.player_y - player_h)

        self._move_pipes()

        self._draw()
        _update_display()
        return Outcome(observation=_get_screen(),
                       terminal=terminal, reward=reward)

    ###########################################################################
    def _draw_bg(self):
        SCREEN.blit(self._images['background'], (0, 0))

    def _draw_pipes(self, upperPipes, lowerPipes):
        for u, l in zip(upperPipes, lowerPipes):
            SCREEN.blit(self._images['pipe'][0], (u['x'], u['y']))
            SCREEN.blit(self._images['pipe'][1], (l['x'], l['y']))

    def _draw_base(self, x, y):
        SCREEN.blit(self._images['base'], (x, y))

    def _draw_player(self, index, x, y):
        SCREEN.blit(self._images['player'][index], (x, y))

    def _draw_digit(self, digit, x):
        SCREEN.blit(self._images['numbers'][digit], (x, screen_height * 0.1))

    def _draw_score(self, score):
        digits = [int(x) for x in list(str(score))]
        widths = [self._images['numbers'][digit].get_width()
                  for digit in digits]

        x = (screen_width - sum(widths)) / 2
        for digit, width in zip(digits, widths):
            self._draw_digit(digit, x)
            x += width

    def _draw(self):
        self._draw_bg()
        self._draw_pipes(self.upperPipes, self.lowerPipes)
        self._draw_base(self.base_x, self.base_y)
        self._draw_score(self.score)
        self._draw_player(self.motion_index, self.player_x, self.player_y)
