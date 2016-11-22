"""Module for defining luchador command entry pointi"""
from __future__ import absolute_import

import sys
from argparse import ArgumentParser as AP

from . import exercise, serve


def entry_point():
    """Entry point for `luchador` command"""
    parser = AP(
        description='luchador'
    )
    parser.add_argument(
        'subcommand',
        choices=['exercise', 'serve']
    )

    args = parser.parse_args(sys.argv[1:2])
    if args.subcommand == 'exercise':
        exercise.entry_point()
    elif args.subcommand == 'serve':
        serve.entry_point()
