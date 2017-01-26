"""Module to define ``luchador serve`` subcommand"""
from __future__ import absolute_import

import logging
import argparse

import luchador.util
import luchador.env.remote

_LG = logging.getLogger(__name__)


def _start_server(app, port):
    server = luchador.env.remote.create_server(app, port=port)
    _LG.info('Starting server on port %d', port)
    try:
        server.start()
    except KeyboardInterrupt:
        server.stop()
    except BaseException:
        _LG.exception('Unexpected error on server at port %d', port)
        server.stop()
    _LG.info('Server on port %d stopped.', port)


###############################################################################
def _parse_command_line_arguments():
    parser = argparse.ArgumentParser(
        description='Start environment server',
    )
    parser.add_argument(
        'serve',
        help='Start server'
    )
    parser.add_argument(
        'type',
        help='The server type to launch', choices=['env', 'manager'])
    parser.add_argument(
        '--environment',
        help='YAML file contains environment configuration')
    parser.add_argument(
        '--port', type=int, default=5000,
        help='Port number to run server')
    return parser.parse_known_args()[0]


def entry_point():
    """Entry porint for `luchador exercise` command"""
    args = _parse_command_line_arguments()
    if args.type == 'env':
        if args.environment is None:
            raise ValueError('Environment config is not given')
        env_config = luchador.util.load_config(args.environment)
        env = luchador.env.get_env(env_config['name'])(**env_config['args'])
        app = luchador.env.remote.create_env_app(env)
    else:
        app = luchador.env.remote.create_manager_app()

    _start_server(app, args.port)
