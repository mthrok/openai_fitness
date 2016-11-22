from __future__ import absolute_import

import logging
import argparse

import luchador.util
import luchador.env.server

_LG = logging.getLogger(__name__)


def _start_server(env_config, port):
    env = luchador.env.get_env(env_config['name'])(**env_config['args'])
    app = luchador.env.server.create_app(env)
    server = luchador.env.server.create_server(app, port=port)
    try:
        _LG.info('Starting environment on port %d', port)
        server.start()
    except KeyboardInterrupt:
        _LG.info('Stopping environment server')
        server.stop()
    except BaseException:
        _LG.exception('Stopping environment server')
        server.stop()


###############################################################################
def _parse_command_line_arguments():
    parser = argparse.ArgumentParser(
        description='Start environment server',
    )
    parser.add_argument('server')  # Placeholder for subcommanding
    parser.add_argument(
        'environment',
        help='YAML file contains environment configuration')
    parser.add_argument(
        '--port', type=int, default=5000,
        help='Port number to run environment')
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()


def _set_logging_level(debug):
    level = logging.DEBUG if debug else logging.INFO
    logging.getLogger('luchador').setLevel(level)


def entry_point():
    """Entry porint for `luchador exercise` command"""
    args = _parse_command_line_arguments()
    env_config = luchador.util.load_config(args.environment)
    _set_logging_level(args.debug)
    _start_server(env_config, args.port)
