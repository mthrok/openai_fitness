from __future__ import absolute_import

from itertools import cycle

import pygame
import numpy as np
from scipy.misc import imresize

from luchador.common import pprint_dict
from . import fb_util as util
from .fb_component import Ground, Background, Pipes, Player
from ..base import BaseEnvironment, Outcome


def get_index_generator(repeat=5, pattern=None):
    indices = []
    pattern = pattern if pattern else [0, 1, 2, 1]
    for val in pattern:
        indices.extend([val] * repeat)
    return cycle(indices)


def pixel_collides(comp1, comp2, hitmask1, hitmask2):
    rect1 = pygame.Rect(comp1.x, comp1.y, comp1.w, comp1.h)
    rect2 = pygame.Rect(comp2.x, comp2.y, comp2.w, comp2.h)

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
    def __init__(
            self, repeat_action=4, random_seed=None,
            width=288, height=512, grayscale=False,
            fastforward=False, play_sound=True):
        self.repeat_action = repeat_action
        self.random_seed = random_seed
        self.width = width
        self.height = height
        self.grayscale = grayscale
        self.play_sound = play_sound
        self.fastforward = fastforward

        self.screen_width = 288
        self.screen_height = 512
        self.rng = np.random.RandomState(seed=random_seed)
        self.player_motion_index = 0
        self.total_frame_number = 0
        self.episode_frame_number = 0

        self._init_pygame()
        self._load_assets()
        self._init_components()

        self._get_screen = (self._get_screen_grayscale if self.grayscale else
                            self._get_screen_rgb)
        if height == self.screen_height and width == self.screen_width:
            self.resize = None
        else:
            self.resize = ((height, width) if self.grayscale else
                           (height, width, 3))

    def _init_pygame(self):
        screen_size = (self.screen_width, self.screen_height)
        pygame.init()
        self.fps = 1000 if self.fastforward else 30
        self.fps_clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode(screen_size)
        pygame.display.set_caption('Flappy Bird')

    def _load_assets(self):
        self._sounds = util.load_sounds()
        self._sprites = util.load_sprites()

    def _init_components(self):
        self.bg = Background(w=self.screen_width, h=self.screen_height)
        self._init_ground()
        self._init_pipes()
        self._init_player()

    def _init_ground(self):
        w, h = self._sprites['ground'].get_size()
        y = self.screen_height * 0.79
        vx = -100
        shift = - w + self.screen_width
        self.ground = Ground(w, h, y, vx, shift)

    def _init_pipes(self):
        vx, y_gap = -4, 100
        w, h = self._sprites['pipes'][0]['images'][0].get_size()
        x_gap = self.screen_width / 2
        y_min = int(0.2 * self.ground.y - h)
        y_max = int(0.8 * self.ground.y - h - y_gap)
        self.pipes = Pipes(w, h, vx, y_min, y_max, y_gap, x_gap,
                           n_pipes=3, rng=self.rng)

    def _init_player(self):
        w, h = self._sprites['players'][0]['images'][0].get_size()
        x = int(0.2 * self.screen_width)
        y = int((self.screen_height - h)/2)
        y_max = self.ground.y - h
        vy_flap = vy = -9
        vy_max = 10
        ay = 1
        self.player = Player(w, h, x, y, y_max, vy, vy_flap, vy_max, ay)

    @property
    def n_actions(self):
        return 2

    ###########################################################################
    def reset(self):
        self.score = 0
        self.episode_frame_number = 0
        self.player_motion_indices = get_index_generator()
        self.reset_color()
        self.bg.reset()
        self.ground.reset()
        self.pipes.reset()
        self.player.reset()
        self._draw()
        return Outcome(observation=self._get_observation(),
                       terminal=False, reward=0, state=self._get_state())

    def reset_color(self):
        sprites = self._sprites
        i = self.rng.randint(len(sprites['bgs']))
        sprites['bg'] = sprites['bgs'][i]
        i = self.rng.randint(len(sprites['pipes']))
        sprites['pipe'] = sprites['pipes'][i]
        i = self.rng.randint(len(sprites['players']))
        sprites['player'] = sprites['players'][i]

    def _get_state(self):
        return {
            'total_frame_number': self.total_frame_number,
            'episode_frame_number': self.episode_frame_number,
        }

    ###########################################################################
    def step(self, tapped):
        reward, terminal = 0, False
        for _ in range(self.repeat_action):
            r, t = self._step(tapped)
            reward += r
            terminal = terminal or t
            if terminal:
                break
        return Outcome(observation=self._get_observation(), terminal=terminal,
                       reward=reward, state=self._get_state())

    def _step(self, tapped):
        self.player_motion_index = self.player_motion_indices.next()
        self.total_frame_number += 1
        self.episode_frame_number += 1

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
            reward = 1 if self._passed() else 0
            terminal = False

        if reward:
            self._play_sound('point')
        self.score += reward

        self._draw()
        return reward, terminal

    def _passed(self):
        player_cx = self.player.cx
        player_cy = self.player.cy
        for t, b in self.pipes.pipes:
            pipe_cx = t.cx
            gap_top, gap_bottom = t.y + t.h, b.h
            if (
                    pipe_cx <= player_cx < pipe_cx + 4 and
                    gap_top <= player_cy < gap_bottom
            ):
                return True
        return False

    def _crashed(self):
        if self.player.b >= self.ground.y:
            return True  # crashed into the ground

        p_mask = self._sprites['player']['hitmasks'][self.player_motion_index]
        t_mask, b_mask = self._sprites['pipe']['hitmasks']
        for t, b in self.pipes.pipes:
            if(
                    pixel_collides(self.player, t, p_mask, t_mask) or
                    pixel_collides(self.player, b, p_mask, b_mask)
            ):
                return True
        return False

    ###########################################################################
    def _draw(self):
        self._draw_bg()
        self._draw_pipes()
        self._draw_ground()
        self._draw_score()
        self._draw_player()
        self._update_display()

    def _draw_screen(self, image, x, y):
        self.screen.blit(image, (x, y))

    def _draw_bg(self):
        self._draw_screen(self._sprites['bg'], 0, 0)

    def _draw_pipes(self):
        t_image, b_image = self._sprites['pipe']['images']
        for u, l in self.pipes.pipes:
            self._draw_screen(t_image, u.x, u.y)
            self._draw_screen(b_image, l.x, l.y)

    def _draw_ground(self):
        ground = self._sprites['ground']
        self._draw_screen(ground, self.ground.x, self.ground.y)

    def _draw_score(self):
        digit_images = self._sprites['digits']
        digits = [int(x) for x in list(str(self.score))]
        widths = [digit_images[d].get_width() for d in digits]

        x = (self.screen_width - sum(widths)) / 2
        y = self.screen_height * 0.1
        for d, width in zip(digits, widths):
            self._draw_screen(digit_images[d], x, y)
            x += width

    def _draw_player(self):
        image = self._sprites['player']['images'][self.player_motion_index]
        self._draw_screen(image, self.player.x, self.player.y)

    def _update_display(self):
        pygame.display.update()
        self.fps_clock.tick(self.fps)

    ###########################################################################
    def _play_sound(self, key):
        if self.play_sound:
            self._sounds[key].play()

    def _get_screen_rgb(self):
        return pygame.surfarray.array3d(self.screen).transpose(1, 0, 2)

    def _get_screen_grayscale(self):
        return self._get_screen_rgb().mean(axis=2)

    def _get_observation(self):
        screen = self._get_screen()
        if self.resize:
            return imresize(screen, self.resize)
        return screen

    ###########################################################################
    def __repr__(self):
        return pprint_dict({
            self.__class__.__name__: {
                'repeat_action': self.repeat_action,
                'random_seed': self.random_seed,
                'width': self.width,
                'height': self.height,
                'grayscale': self.grayscale,
                'play_sound': self.play_sound,
                'fastforward': self.fastforward,
            }
        })
