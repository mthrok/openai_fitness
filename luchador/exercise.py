from __future__ import absolute_import

import logging
import datetime

from .env import get_env
from .agent import get_agent
from .util import load_config
from .episode_runner import EpisodeRunner

_LG = logging.getLogger(__name__)


###############################################################################
def main(env, agent, episodes, steps, debug=False):
    if debug:
        logging.getLogger('luchador').setLevel(logging.DEBUG)

    Environment = get_env(env['name'])
    env = Environment(**env['args'])
    _LG.info('\n{}'.format(env))

    Agent = get_agent(agent['name'])
    agent = Agent(**agent['args'])
    agent.set_env_info(env)
    agent.init()
    _LG.info('\n{}'.format(agent))

    runner = EpisodeRunner(env, agent)
    for i in range(episodes):
        _LG.debug('Running episode: {}'.format(i))
        runner.run_episode(steps)
    _LG.info('Done')


###############################################################################
def _parse_command_line_arguments():
    import argparse
    ap = argparse.ArgumentParser(
        description='Run environment with agents.',
    )
    ap.add_argument(
        'env_config',
        help='YAML file contains environment configuration')
    ap.add_argument(
        'agent_config',
        help='YAML file contains agent configuration')
    ap.add_argument('--episodes', '-ep', type=int, default=1000)
    ap.add_argument('--steps', '-ts', type=int, default=10000)
    ap.add_argument('--debug', action='store_true')
    return ap.parse_args()


def _get_current_time():
    return datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')


def _parse_config():
    args = _parse_command_line_arguments()
    config = {
        'env': load_config(args.env_config),
        'agent': load_config(args.agent_config),
        'episodes': args.episodes,
        'steps': args.steps,
        'debug': args.debug,
    }
    return config


def entry_point():
    main(**_parse_config())
