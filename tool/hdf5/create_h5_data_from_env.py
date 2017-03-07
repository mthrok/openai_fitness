#!/usr/bin/env python
"""Run environment and save the state in HDF5"""
from __future__ import print_function
from __future__ import absolute_import

import h5py
import numpy as np

from luchador.env import get_env
from luchador.util import load_config


def _parse_command_line_args():
    from argparse import ArgumentParser as AP
    ap = AP(
        description='Create state data from environment'
    )
    ap.add_argument('env', help='YAML file contains environment config')
    ap.add_argument('output', help='Output HDF5 file name')
    ap.add_argument('key', help='Name of dataset in the output file')
    ap.add_argument('--channel', type=int, default=4)
    ap.add_argument('--batch', type=int, default=32)
    return ap.parse_args()


def create_env(cfg_file):
    """Load Environment config file and instantiate"""
    cfg = load_config(cfg_file)
    env = get_env(cfg['name'])(**cfg['args'])
    print('\n{}'.format(env))
    return env


def _create_data(env, channel, batch):
    samples = []
    env.reset()
    for _ in range(batch):
        sample = []
        for _ in range(channel):
            outcome = env.step(0)
            sample.append(outcome.observation)
            if outcome.terminal:
                env.reset()
        samples.append(sample)
    return np.asarray(samples, dtype=np.uint8)


def _save(data, output_file, key='data'):
    file_ = h5py.File(output_file, 'a')
    if key in file_:
        del file_[key]
    file_.create_dataset(key, data=data)
    file_.close()


def _main():
    args = _parse_command_line_args()
    env = create_env(args.env)
    data = _create_data(env, args.channel, args.batch)
    _save(data, args.output, args.key)


if __name__ == '__main__':
    _main()
