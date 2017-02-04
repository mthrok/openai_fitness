"""Script to run env-agent scheme as independant program"""
from __future__ import division
from __future__ import absolute_import

import logging

from luchador.env import get_env
from luchador.agent import get_agent
from luchador.util import load_config
from luchador.episode_runner import EpisodeRunner

_LG = logging.getLogger(__name__)


###############################################################################
def _main(env, agent, episodes, steps, report_every=1000):
    agent.init(env)
    runner = EpisodeRunner(env, agent, max_steps=steps)

    _LG.info('Running %s episodes', episodes)
    n_ep, time_, steps_, rewards_ = 0, 0, 0, 0.0
    for i in range(1, episodes+1):
        stats = runner.run_episode()

        n_ep += 1
        time_ += stats['time']
        steps_ += stats['steps']
        rewards_ += stats['rewards']
        if i % report_every == 0 or i == episodes:
            _LG.info('Finished episode: %d', i)
            _LG.info('  Rewards:     %12.3f', rewards_)
            _LG.info('               %12.3f [/epi]', rewards_ / n_ep)
            _LG.info('               %12.3f [/steps]', rewards_ / steps_)
            _LG.info('  Steps:       %8d', steps_)
            _LG.info('               %12.3f [/epi]', steps_ / n_ep)
            _LG.info('               %12.3f [/sec]', steps_ / time_)
            _LG.info('  Total Steps: %8d', runner.steps)
            _LG.info('  Total Time:  %s', _format_time(runner.time))
            n_ep, time_, steps_, rewards_ = 0, 0, 0, 0.
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
def _load_additional_sources(*files):
    for file_ in files:
        _LG.info('Loading additional source file: %s', file_)
        exec('import {}'.format(file_))  # pylint: disable=exec-used


def _make_agent(config_file):
    config = (
        load_config(config_file) if config_file else
        {'typename': 'NoOpAgent', 'args': {}}
    )
    return get_agent(config['typename'])(**config.get('args', {}))


def _make_env(config_file, host, port):
    config = load_config(config_file)
    if config['typename'] == 'RemoteEnv':
        if port:
            config['args']['port'] = port
        if host:
            config['args']['host'] = host
    return get_env(config['typename'])(**config.get('args', {}))


def entry_point(args):
    """Entry porint for `luchador exercise` command"""
    env = _make_env(args.environment, args.host, args.port)
    agent = _make_agent(args.agent)

    _LG.info('\n%s', env)
    _LG.info('\n%s', agent)

    try:
        _main(
            env, agent, episodes=args.episodes,
            steps=args.steps, report_every=args.report)
    finally:
        if env.__class__.__name__ == 'RemoteEnv' and args.kill:
            _LG.info('Killing environment server')
            success = env.kill()
            _LG.info(
                'Environment %s',
                'killed' if success else 'failed to terminate'
            )
