from __future__ import absolute_import

from itertools import cycle

import pygame
import numpy as np

from . import util
from ..base import BaseEnvironment, Outcome


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


class Background(object):
    def __init__(self, game):
        self.game = game
        self.images = util.load_backgrounds()

    def reset(self):
        self.index = self.game.rng.randint(len(self.images))

    @property
    def image(self):
        return self.images[self.index]

    @property
    def width(self):
        return self.image.get_width()


class Ground(object):
    def __init__(self, game):
        self.game = game
        self.image = util.load_ground()

    def reset(self):
        self.x = 0
        self.y = self.game.screen_height * 0.79
        self.x_shift = self.width - self.game.bg.width

    def update(self):
        self.x = (self.x - 100) % (-self.x_shift)

    @property
    def width(self):
        return self.image.get_width()


class Player(object):
    def __init__(self, game):
        self.game = game
        self._images, self._hitmasks = util.load_player()

        # Constants
        self.g = 1            # Downward accleration
        self.v_max = 10       # Max downward velocity
        self.v_flapped = -9   # Velosity when flapped
        self.x = int(0.2 * self.game.screen_width)

    def reset(self):
        self.motion_indices = get_index_generator(repeat=3)
        self.motion_index = self.motion_indices.next()
        self.color_index = self.game.rng.randint(len(self._images))

        self.v = -9
        self.y = int((self.game.screen_height - self.height) / 2)

    def update(self, tapped):
        self.motion_index = self.motion_indices.next()
        flapped = False
        if tapped:
            if self.y >= 0:
                self.v = self.v_flapped
                flapped = True
        elif self.v < self.v_max:
            self.v += self.g

        self.y += self.v
        self.y = min(self.y, self.game.ground.y - self.height)
        return flapped

    def get_rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)

    @property
    def images(self):
        return self._images[self.color_index]

    @property
    def image(self):
        return self.images[self.motion_index]

    @property
    def hitmask(self):
        return self._hitmasks[self.color_index][self.motion_index]

    @property
    def width(self):
        return self.image.get_width()

    @property
    def height(self):
        return self.image.get_height()


class Pipes(object):
    def __init__(self, game):
        self.game = game
        self._images, self._hitmasks = util.load_pipes()

        # Constants
        self.v = -4     # Pipe moving speed
        self.gap = 100  # Gap between upper and lower pipes

    def reset(self):
        self.color_index = self.game.rng.randint(1, len(self._images)) - 1

        screen_width = self.game.screen_width
        self.upper, self.lower = [], []
        for i in range(2):
            upper, lower = self._get_new_pipe_position()
            x = screen_width + 200 + (i * screen_width / 2)
            self.upper.append({'x': x, 'y': upper['y']})
            self.lower.append({'x': x, 'y': lower['y']})

    def _get_new_pipe_position(self):
        ground_y = self.game.ground.y
        x = self.game.screen_width + 10
        min_y = int(ground_y * 0.2)
        max_y = int(ground_y * 0.8 - self.gap)
        base_y = self.game.rng.randint(min_y, max_y)  # Top of gap

        upper_pipe = {'x': x, 'y': base_y - self.height}
        lower_pipe = {'x': x, 'y': base_y + self.gap}
        return upper_pipe, lower_pipe

    def update(self):
        # move pipes to left
        for upper, lower in zip(self.upper, self.lower):
            upper['x'] += self.v
            lower['x'] += self.v
        # add new pipe when first pipe is about to touch left of screen
        if 0 < self.upper[0]['x'] < 5:
            upper, lower = self._get_new_pipe_position()
            self.upper.append(upper)
            self.lower.append(lower)
        # remove first pipe if its out of the screen
        if self.upper[0]['x'] < -self.width:
            self.upper.pop(0)
            self.lower.pop(0)

    @property
    def images(self):
        return self._images[self.color_index]

    @property
    def hitmasks(self):
        return self._hitmasks[self.color_index]

    @property
    def width(self):
        return self.images[0].get_width()

    @property
    def height(self):
        return self.images[0].get_height()


class FlappyBird(BaseEnvironment):
    def __init__(self, random_seed=None, play_sound=False):
        # Constants
        self.fps = 30
        self.screen_width = 288
        self.screen_height = 512
        self.rng = np.random.RandomState(seed=random_seed)
        self.sound_enabled = play_sound

        self._init_pygame()
        self._load_assets()

        self.bg = Background(self)
        self.ground = Ground(self)
        self.pipes = Pipes(self)
        self.player = Player(self)

    def _init_pygame(self):
        screen_size = (self.screen_width, self.screen_height)
        pygame.init()
        self.fps_clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode(screen_size)
        pygame.display.set_caption('Flappy Bird')

    def _play_sound(self, key):
        if self.sound_enabled:
            self._sounds[key].play()

    def _load_assets(self):
        self._digits = util.load_digits()
        self._sounds = util.load_sounds()

    @property
    def n_actions(self):
        return 2

    ###########################################################################
    def reset(self):
        self.score = 0
        self.bg.reset()
        self.ground.reset()
        self.pipes.reset()
        self.player.reset()
        self._draw()
        obs = self._get_screen()
        return Outcome(observation=obs, terminal=False, reward=0)

    ###########################################################################
    def step(self, tapped):
        self.ground.update()
        self.pipes.update()
        flapped = self.player.update(tapped)

        if flapped:
            self._play_sound('wing')

        if self._crashed():
            self._play_sound('hit')
            reward = 0
            terminal = True
        else:
            reward = self._get_reward()
            terminal = False
        if reward:
            self._play_sound('point')
        self.score += reward

        self._draw()
        return Outcome(observation=self._get_screen(),
                       terminal=terminal, reward=reward)

    def _get_reward(self):
        pipe_w = self.pipes.width
        player_x = self.player.x + self.player.width / 2
        for pipe in self.pipes.upper:
            pipe_x = pipe['x'] + pipe_w / 2
            if pipe_x <= player_x < pipe_x + 4:
                return 1
        return 0

    def _crashed(self):
        if self.player.y + self.player.height + 1 >= self.ground.y:
            return True  # crashed into the ground

        p_width, p_height = self.pipes.width, self.pipes.height
        u_mask, l_mask = self.pipes.hitmasks
        p_mask = self.player.hitmask
        p_rect = self.player.get_rect()
        for u, l in zip(self.pipes.upper, self.pipes.lower):
            u_pipe = pygame.Rect(u['x'], u['y'], p_width, p_height)
            l_pipe = pygame.Rect(l['x'], l['y'], p_width, p_height)
            if(
                    pixel_collides(p_rect, u_pipe, p_mask, u_mask) or
                    pixel_collides(p_rect, l_pipe, p_mask, l_mask)
            ):
                return True
        return False

    ###########################################################################
    def _draw_bg(self):
        self._draw_screen(self.bg.image, 0, 0)

    def _draw_screen(self, image, x, y):
        self.screen.blit(image, (x, y))

    def _draw_pipes(self):
        for u, l in zip(self.pipes.upper, self.pipes.lower):
            self._draw_screen(self.pipes.images[0], u['x'], u['y'])
            self._draw_screen(self.pipes.images[1], l['x'], l['y'])

    def _draw_ground(self):
        self._draw_screen(self.ground.image, self.ground.x, self.ground.y)

    def _draw_player(self):
        self._draw_screen(self.player.image, self.player.x, self.player.y)

    def _draw_score(self):
        digits = [int(x) for x in list(str(self.score))]
        widths = [self._digits[d].get_width() for d in digits]

        x = (self.screen_width - sum(widths)) / 2
        y = self.screen_height * 0.1
        for d, width in zip(digits, widths):
            self._draw_screen(self._digits[d], x, y)
            x += width

    def _update_display(self):
        pygame.display.update()
        self.fps_clock.tick(self.fps)

    def _draw(self):
        self._draw_bg()
        self._draw_pipes()
        self._draw_ground()
        self._draw_score()
        self._draw_player()
        self._update_display()

    def _get_screen(self):
        return pygame.surfarray.array3d(pygame.display.get_surface())
