from __future__ import division
from __future__ import absolute_import

import logging

from .env import get_env
from .agent import get_agent
from .util import load_config
from .episode_runner import EpisodeRunner

_LG = logging.getLogger(__name__)


###############################################################################
def main(env, agent, episodes, steps, report_every=1000):
    Environment = get_env(env['name'])
    env = Environment(**env['args'])
    _LG.info('\n{}'.format(env))

    Agent = get_agent(agent['name'])
    agent = Agent(**agent['args'])
    _LG.info('\n{}'.format(agent))
    agent.init(env)

    runner = EpisodeRunner(env, agent, max_steps=steps)

    _LG.info('Running {} episodes'.format(episodes))
    n_ep, time_, steps_, rewards = 0, 0, 0, 0.0
    for i in range(1, episodes+1):
        stats = runner.run_episode()

        n_ep += 1
        time_ += stats['time']
        steps_ += stats['steps']
        rewards += stats['rewards']
        if i % report_every == 0 or i == episodes:
            _LG.info('Finished episode: {}'.format(i))
            _LG.info('  Rewards:     {:8.1f} [/epi]'.format(rewards / n_ep))
            _LG.info('  Steps:       {:8.1f} [/epi]'.format(steps_ / n_ep))
            _LG.info('               {:8.1f} [/sec]'.format(steps_ / time_))
            _LG.info('  Total Steps: {:8d}'.format(runner.steps))
            _LG.info('  Total Time:  {:8.1f} [sec]'.format(runner.time))
            n_ep, time_, steps_, rewards = 0, 0, 0, 0.0
    _LG.info('Done')


###############################################################################
def _parse_command_line_arguments():
    import argparse
    ap = argparse.ArgumentParser(
        description='Run environment with agents.',
    )
    ap.add_argument(
        'environment',
        help='YAML file contains environment configuration')
    ap.add_argument(
        '--agent',
        help='YAML file contains agent configuration')
    ap.add_argument(
        '--episodes', type=int, default=1000,
        help='Run this number of episodes.')
    ap.add_argument(
        '--report', type=int, default=1000,
        help='Report run stats every this number of episodes')
    ap.add_argument(
        '--steps', type=int, default=10000,
        help='Set max steps for each episode.')
    ap.add_argument(
        '--sources', nargs='+', default=[],
        help='Source files that contain custom Agent/Env etc')
    ap.add_argument('--debug', action='store_true')
    return ap.parse_args()


def _set_logging_level(debug):
    logging.getLogger('luchador').setLevel(
        logging.DEBUG if debug else logging.INFO)


def _load_additional_sources(*files):
    for f in files:
        _LG.info('Loading additional source file: {}'.format(f))
        exec('import {}'.format(f))


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
    main(**_parse_config())
