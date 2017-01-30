"""Module for defining luchador command entry point"""
from __future__ import absolute_import

import logging

from . import parser


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
    args = parser.parse_command_line_args()
    _initialize_logger(args.debug)
    args.func(args)
