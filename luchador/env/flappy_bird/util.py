from __future__ import absolute_import

import os
import sys
import pygame

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
    mask = []
    for x in range(image.get_width()):
        mask.append([])
        for y in range(image.get_height()):
            mask[x].append(bool(image.get_at((x, y))[3]))
    return mask


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


def load_player():
    images = [
        [
            _load_sprite('{color}bird-{direction}flap.png'
                         .format(color=c, direction=d)).convert_alpha()
            for d in ['up', 'mid', 'down']
        ] for c in ['red', 'blue', 'yellow']
    ]
    hitmasks = [
        [_gen_hitmask(img) for img in imgs] for imgs in images
    ]
    return images, hitmasks


def load_pipes():
    files = ['pipe-green.png', 'pipe-red.png']
    images = [
        [pygame.transform.rotate(pipe, 180), pipe]
        for pipe in map(lambda f: _load_sprite(f).convert_alpha(), files)
    ]
    hitmasks = [
        [_gen_hitmask(img) for img in imgs] for imgs in images
    ]
    return images, hitmasks
