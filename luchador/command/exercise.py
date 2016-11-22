"""Script to run env-agent scheme as independant program"""
from __future__ import division
from __future__ import absolute_import

import logging
import argparse

from luchador.env import get_env
from luchador.agent import get_agent
from luchador.util import load_config
from luchador.episode_runner import EpisodeRunner

_LG = logging.getLogger(__name__)


###############################################################################
def _main(env, agent, episodes, steps, report_every=1000):
    env = get_env(env['name'])(**env['args'])
    agent = get_agent(agent['name'])(**agent['args'])
    _LG.info('\n%s', env)
    _LG.info('\n%s', agent)

    agent.init(env)
    runner = EpisodeRunner(env, agent, max_steps=steps)

    _LG.info('Running %s episodes', episodes)
    n_ep, time_, steps_, rewards = 0, 0, 0, 0.0
    for i in range(1, episodes+1):
        stats = runner.run_episode()

        n_ep += 1
        time_ += stats['time']
        steps_ += stats['steps']
        rewards += stats['rewards']
        if i % report_every == 0 or i == episodes:
            _LG.info('Finished episode: %d', i)
            _LG.info('  Rewards:     %12.3f [/epi]', rewards / n_ep)
            _LG.info('  Steps:       %8d', steps_)
            _LG.info('               %12.3f [/epi]', steps_ / n_ep)
            _LG.info('               %12.3f [/sec]', steps_ / time_)
            _LG.info('  Total Steps: %8d', runner.steps)
            _LG.info('  Total Time:  %s', _format_time(runner.time))
            n_ep, time_, steps_, rewards = 0, 0, 0, 0.0
    _LG.info('Done')


def _format_time(seconds):
    """Format duration in second to readable expression"""
    min_, sec = divmod(seconds, 60)
    hour, min_ = divmod(min_, 60)
    day, hour = divmod(hour, 24)

    ret = '{:02d}:{:02d}:{:02d}'.format(int(hour), int(min_), int(sec))
    if day:
        ret = '{:d} Days {}'.format(int(day), ret)
    return ret


###############################################################################
def _parse_command_line_arguments():
    parser = argparse.ArgumentParser(
        description='Run environment with agents.',
    )
    parser.add_argument('exercise')  # Placeholder for subcommanding
    parser.add_argument(
        'environment',
        help='YAML file contains environment configuration')
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
        '--sources', nargs='+', default=[],
        help='Source files that contain custom Agent/Env etc')
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()


def _set_logging_level(debug):
    logging.getLogger('luchador').setLevel(
        logging.DEBUG if debug else logging.INFO)


def _load_additional_sources(*files):
    for file_ in files:
        _LG.info('Loading additional source file: %s', file_)
        exec('import {}'.format(file_))  # pylint: disable=exec-used


def _parse_config():
    args = _parse_command_line_arguments()

    _set_logging_level(args.debug)
    _load_additional_sources(*args.sources)

    config = {
        'env': load_config(args.environment),
        'agent': (load_config(args.agent) if args.agent else
                  {'name': 'NoOpAgent', 'args': {}}),
        'episodes': args.episodes,
        'steps': args.steps,
        'report_every': args.report,
    }
    return config


def entry_point():
    """Entry porint for `luchador exercise` command"""
    _main(**_parse_config())
