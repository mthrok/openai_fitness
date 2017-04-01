"""Module for defining luchador command entry point"""
from __future__ import absolute_import

from argparse import ArgumentParser as AP


def _append_environment(parser):
    parser.add_argument(
        'environment',
        help='YAML file contains environment configuration'
    )


def _append_debug_flag(parser):
    parser.add_argument('--debug', action='store_true')


def _append_port_number(parser, default):
    parser.add_argument(
        '--port', type=int, default=default,
        help='Port number to run server. Default: {}'.format(default),
    )


def _add_exercise_subcommand_parser(subparsers):
    """Define command line arguments for `luchador exercise` command"""
    from . import exercise
    parser = subparsers.add_parser(
        'exercise',
        help='Run environment and agent as a single program')
    _append_environment(parser)
    parser.add_argument(
        '--agent',
        help='YAML file contains agent configuration')
    parser.add_argument(
        '--episodes', type=int, default=1000,
        help='Run this number of episodes.')
    parser.add_argument(
        '--report', type=int, default=1000,
        help='Report run stats every this number of episodes')
    parser.add_argument(
        '--steps', type=int, default=10000,
        help='Set max steps for each episode.')
    parser.add_argument(
        '--save-dir', help='Save environment transition.')
    parser.add_argument(
        '--sources', nargs='+', default=[],
        help='Source files that contain custom Agent/Env etc')
    parser.add_argument(
        '--port',
        help='If environment is RemoteEnv, overwrite port number')
    parser.add_argument(
        '--host',
        help='If environment is RemoteEnv, overwrite host')
    parser.add_argument(
        '--kill', action='store_true',
        help='If environment is RemoveEnv, kill server after finish')
    _append_debug_flag(parser)
    parser.set_defaults(func=exercise.entry_point)


def _add_serve_subcommand_parser(subparsers):
    """Define command line arguments for `luchador serve` command"""
    from . import serve
    subparser = subparsers.add_parser(
        'serve',
        help='Start environment server or manager server')
    subsubparsers = subparser.add_subparsers()

    env_parser = subsubparsers.add_parser(
        'env',
        description='Start environment server',
    )
    _append_environment(env_parser)
    _append_port_number(env_parser, 5000)
    _append_debug_flag(env_parser)
    env_parser.set_defaults(func=serve.entry_point_env)

    man_parser = subsubparsers.add_parser(
        'manager',
        description='Start manager server',
    )
    _append_debug_flag(man_parser)
    _append_port_number(man_parser, 5001)
    man_parser.set_defaults(func=serve.entry_point_manager)


def parse_command_line_args():
    """Parser command line arguments for luchador"""
    parser = AP(
        description='luchador command line tools'
    )
    subparsers = parser.add_subparsers()
    _add_exercise_subcommand_parser(subparsers)
    _add_serve_subcommand_parser(subparsers)
    return parser.parse_args()
