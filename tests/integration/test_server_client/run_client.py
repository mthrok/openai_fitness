"""Test each Environment server route"""
from __future__ import absolute_import

import logging
import argparse

import numpy as np

import luchador.env.client

_LG = logging.getLogger()


def _run_client(port, host):
    client = luchador.env.client.EnvClient(port=port, host=host)

    _LG.info('Getting the last Outcome')
    outcome = client.outcome()

    _LG.info('Getting #Actions from Environment')
    n_actions = client.n_actions()
    _LG.info('%d', n_actions)

    _LG.info('Running:')
    while not outcome.terminal:
        action = np.random.randint(n_actions)
        outcome = client.step(action)

    _LG.info('Resetting Environment')
    outcome = client.reset()
    _LG.info('%s', outcome)

    _LG.info('Kiling the server')
    _LG.info('%s', 'Success' if client.kill() else 'Failed')


def _parse_command_line_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=5000)
    ap.add_argument('--debug', action='store_true')
    return ap.parse_args()


def _main():
    args = _parse_command_line_arguments()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    _run_client(args.port, args.host)


if __name__ == '__main__':
    _main()
