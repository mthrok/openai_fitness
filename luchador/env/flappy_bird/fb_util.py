from __future__ import absolute_import

import os
import sys

import pygame
import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
_ASSET_DIR = os.path.join(_DIR, 'assets')
_SPRITE_DIR = os.path.join(_ASSET_DIR, 'sprites')
_AUDIO_DIR = os.path.join(_ASSET_DIR, 'audio')


def _load_sound(filename):
    ext = 'wav' if 'win' in sys.platform else 'ogg'
    filename = '{}.{}'.format(filename, ext)
    return pygame.mixer.Sound(os.path.join(_AUDIO_DIR, filename))


def _load_sprite(filename):
    return pygame.image.load(os.path.join(_SPRITE_DIR, filename))


def _gen_hitmask(image):
    w, h = image.get_size()
    mask = np.zeros((w, h), dtype=np.bool)
    for x in range(w):
        for y in range(h):
            mask[x, y] = (bool(image.get_at((x, y))[3]))
    return mask


###############################################################################
def load_sounds():
    return {key: _load_sound(key)
            for key in ['die', 'hit', 'point', 'wing']}


def load_digits():
    return [
            _load_sprite('{digit:1d}.png'.format(digit=d)).convert_alpha()
            for d in range(10)
    ]


def load_backgrounds():
    return [_load_sprite('background-{}.png'.format(s)).convert()
            for s in ['day', 'night']]


def load_ground():
    return _load_sprite('ground.png').convert_alpha()


def load_players():
    ret = []
    for c in ['red', 'blue', 'yellow']:
        images = [
            _load_sprite('{color}bird-{direction}flap.png'
                         .format(color=c, direction=d)).convert_alpha()
            for d in ['up', 'mid', 'down']
        ]
        hitmasks = [_gen_hitmask(image) for image in images]
        ret.append({'images': images, 'hitmasks': hitmasks})
    return ret


def load_pipes():
    ret = []
    for f in ['pipe-green.png', 'pipe-red.png']:
        pipe = _load_sprite(f).convert_alpha()
        images = [pygame.transform.rotate(pipe, 180), pipe]
        hitmasks = [_gen_hitmask(image) for image in images]
        ret.append({'images': images, 'hitmasks': hitmasks})
    return ret


def load_sprites():
    return {
        'bgs': load_backgrounds(),
        'ground': load_ground(),
        'pipes': load_pipes(),
        'players': load_players(),
        'digits': load_digits(),
    }
