import h5py
import numpy as np

from luchador.env import get_env
from luchador.util import load_config


def parse_command_line_args():
    from argparse import ArgumentParser as AP
    ap = AP(
        Description='Create ALE Environment state data'
    )
    ap.add_argument('env', help='YAML file contains environment config')
    ap.add_argument('output', help='Output HDF5 file name')
    ap.add_argument('key', help='Name of dataset in the output file')
    ap.add_argument('--channel', type=int, default=4)
    ap.add_argument('--batch', type=int, default=32)
    return ap.parse_args()


def create_env(cfg_file):
    cfg = load_config(cfg_file)
    Environment = get_env(cfg['name'])
    env = Environment(**cfg['args'])
    print('\n{}'.format(env))
    return env


def create_data(env, channel, batch):
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


def save(data, output_file, key='data'):
    f = h5py.File(output_file, 'a')
    if key in f:
        del f[key]
    f.create_dataset(key, data=data)
    f.close()


def main():
    args = parse_command_line_args()
    env = create_env(args.env)
    data = create_data(env, args.channel, args.batch)

    save(data, args.output, args.key)


if __name__ == '__main__':
    main()
