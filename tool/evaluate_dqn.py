"""
This script evaluates the training of DQN, following the original DQN
 paper procedure, which is;

---
`The trained agents were evaluated by playing each game 30 times for up to
 5 min each time with different initial random conditions
---

"""
from __future__ import division

import os
import time
import logging

from luchador.util import load_config
from luchador.env import get_env
from luchador.agent import get_agent
from luchador.nn import SummaryWriter

_LG = logging.getLogger('luchador')


def _parse_command_line_args():
    from argparse import ArgumentParser as AP
    ap = AP(description=(
        'Given a set of DQN parameters, evaluate the performace'
    ))
    ap.add_argument(
        'env', help='Environment configuration file'
    )
    ap.add_argument(
        'agent', help='Agent configuration file'
    )
    ap.add_argument(
        'input_dir', help='Directory where trained parameters are stored'
    )
    ap.add_argument(
        'output_dir', help='Directory where evaluation result is written'
    )
    ap.add_argument(
        '--prefix', default='DQN',
        help='Prefix of trained parameter files'
    )
    ap.add_argument(
        '--timelimit', type=float, default=300,
        help='Time limit of one episode'
    )
    ap.add_argument(
        '--episodes', type=int, default=30,
        help='#Episodes to run'
    )
    return ap.parse_args()


def _load_env(config_file_path):
    config = load_config(config_file_path)
    return get_env(config['name'])(**config['args'])


def _load_agent(config_file_path):
    config = load_config(config_file_path)
    return get_agent(config['name'])(**config['args'])


def _get_parameter_files(dir_path, prefix):
    files = []
    for filename in os.listdir(dir_path):
        if not filename.startswith(prefix):
            continue
        try:
            ite = int(filename.split('.')[0].split('_')[1])
            f_path = os.path.join(dir_path, filename)
            files.append((ite, f_path))
        except ValueError:
            _LG.warn('Failed to parse file name %s.', filename)
    files.sort(key=lambda f: f[0])
    return files


def _run_episode(env, agent, timelimit=300):
    """Run one episode under the given condition

    Args:
      env (Environment): Environment to run
      agent (Agent): Agent to run
      timelimit (int): Time limit of one episode in second
    """
    outcome = env.reset()
    last_observation = outcome.observation
    agent.reset(last_observation)

    t0 = time.time()
    rewards, steps = 0, 0
    while not outcome.terminal:
        action = agent.act(last_observation)
        outcome = env.step(action)

        rewards += outcome.reward
        steps += 1

        agent.observe(action, outcome)

        elapsed = time.time() - t0
        last_observation = outcome.observation

        if elapsed > timelimit:
            _LG.info('  Reached time limit')

    return rewards, steps


def _run_episodes(env, agent, episodes=30, timelimit=300):
    rewards, steps = [], []
    _LG.info('  Running %d Episodes', episodes)
    for i in range(episodes):
        t0 = time.time()
        reward, step = _run_episode(env, agent, timelimit)
        t1 = time.time()
        _LG.info('  Ep %4d: Rewards %8d, Steps %8d, Time %8d [sec]',
                 i+1, reward, step, t1-t0)
        rewards.append(reward)
        steps.append(step)
    return rewards, steps


def _main():
    args = _parse_command_line_args()

    files = _get_parameter_files(args.input_dir, args.prefix)
    env = _load_env(args.env)
    agent = _load_agent(args.agent)
    agent.init(env)

    writer = SummaryWriter(args.output_dir)
    if agent.session.graph:
        writer.add_graph(agent.session.graph)
    writer.register_stats(['Reward', 'Steps'])

    for ite, file_ in files:
        _LG.info('*** Evaluating %s', file_)
        agent.session.load_from_file(file_)
        rewards, steps = _run_episodes(
            env, agent, episodes=args.episodes, timelimit=args.timelimit)
        _LG.info('Average rewards: %s', sum(rewards) / len(rewards))
        _LG.info('Average steps: %s', sum(steps) / len(steps))
        writer.summarize_stats(ite, {'Reward': rewards, 'Steps': steps})


if __name__ == '__main__':
    _main()
