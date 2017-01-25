"""Module for defining luchador command entry point"""
from __future__ import absolute_import

import logging
from argparse import ArgumentParser as AP


def _parse_command_line_args():
    parser = AP(
        description='luchador'
    )
    parser.add_argument(
        'subcommand',
        choices=['exercise', 'serve']
    )
    parser.add_argument('--debug', action='store_true')
    return parser.parse_known_args()[0]


def _initialize_logger(debug):
    from luchador.util import initialize_logger
    message_format = (
        '%(asctime)s: %(levelname)5s: %(funcName)10s: %(message)s'
        if debug else '%(asctime)s: %(levelname)5s: %(message)s'
    )
    level = logging.DEBUG if debug else logging.INFO
    initialize_logger(
        name='luchador', message_format=message_format, level=level)


def entry_point():
    """Entry point for `luchador` command"""
    from . import exercise, serve

    args = _parse_command_line_args()
    _initialize_logger(args.debug)

    if args.subcommand == 'exercise':
        exercise.entry_point()
    elif args.subcommand == 'serve':
        serve.entry_point()
