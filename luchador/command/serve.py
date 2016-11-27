"""Module to define ``luchador serve`` subcommand"""
from __future__ import absolute_import

import logging
import argparse

import luchador.util
import luchador.env.server

_LG = logging.getLogger(__name__)


def _start_env_server(env_config, port, host):
    env = luchador.env.get_env(env_config['name'])(**env_config['args'])
    app = luchador.env.server.create_env_app(env)
    server = luchador.env.server.create_server(app, port=port, host=host)
    _LG.info('Starting environment on port %d', port)
    try:
        server.start()
    except KeyboardInterrupt:
        server.stop()
    except BaseException:
        _LG.exception('Unexpected error on port %d', port)
        server.stop()
    _LG.info('Environment server on port %d stopped.', port)


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
    parser.add_argument(
        '--host', default='0.0.0.0',
        help='Host to run server')
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()


def _set_logging_level(debug):
    level = logging.DEBUG if debug else logging.INFO
    logging.getLogger('luchador').setLevel(level)


def entry_point():
    """Entry porint for `luchador exercise` command"""
    args = _parse_command_line_arguments()
    _set_logging_level(args.debug)

    if args.type == 'env':
        if args.environment is None:
            raise ValueError('Environment config is not given')
        env_config = luchador.util.load_config(args.environment)
        _start_env_server(env_config, args.port, args.host)
    else:
        pass
