"""Launch environment via manager"""
from __future__ import print_function
from __future__ import absolute_import

import argparse

import requests


def _parse_command_line_args():
    parser = argparse.ArgumentParser(
        description='Launch environment via manager'
    )
    parser.add_argument('--env-port')
    parser.add_argument('--man-port')
    return parser.parse_args()


def _main():
    args = _parse_command_line_args()
    res = requests.post(
        'http://localhost:{}/create'.format(args.man_port),
        json={
            'environment': {
                'name': 'ALEEnvironment',
                'args': {
                    'rom': 'breakout.bin',
                    'display_screen': True,
                }
            },
            'port': args.env_port,
            'host': '0.0.0.0',
        }
    )
    if res.status_code == 200:
        print('Environment launched at {}'.format(args.env_port))
    else:
        raise RuntimeError('Failed to launch environment')


if __name__ == '__main__':
    _main()
