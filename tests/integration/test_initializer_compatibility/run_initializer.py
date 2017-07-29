from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os

import h5py
import numpy as np

import luchador.util
from luchador import nn

_BE = luchador.get_nn_backend()


def _parse_command_line_args():
    from argparse import ArgumentParser as AP
    ap = AP(
        description=(
            'Run Initializer and check if the distribution of '
            'the resulting values is desired one')
    )
    ap.add_argument(
        'config', help='YAML file with initializer config and test config'
    )
    ap.add_argument(
        '--output', help='Name of output HDF5 file.'
    )
    ap.add_argument(
        '--key', help='Name of dataset in the output file.', default='input'
    )
    return ap.parse_args()


def _create_initializer(typename, args):
    return nn.fetch_initializer(typename)(**args)


def _transpose_needed(initializer, shape):
    names = ['XavierInitializer', 'KaimingInitializer']
    return (
        len(shape) == 4 and _BE == 'tensorflow' and
        initializer.__class__.__name__ in names
    )


def _run_initializer(initializer, shape):
    if _transpose_needed(initializer, shape):
        # Shape is given in Theano's filter order, which is
        # [#out-channel, #in-channel, height, width].
        # So as to compute fan-in and fan-out correctly in Tensorflow,
        # we reorder this to
        # [height, width, #in-channel, #out-channel],
        shape = [shape[2], shape[3], shape[1], shape[0]]

    variable = nn.make_variable(
        shape=shape, name='input', initializer=initializer)
    session = nn.Session()
    session.initialize()
    value = session.run(outputs=variable, name='test')

    if _transpose_needed(initializer, shape):
        # So as to make the output comarison easy, we revert the oreder.
        shape = [shape[3], shape[2], shape[0], shape[1]]
    return value


def _print_stats(*arrs):
    print('{sum:>10}  {max:>10}  {min:>10}  {mean:>10} {std:>10}'
          .format(sum='sum', max='max', min='min', mean='mean', std='std'))
    for arr in arrs:
        sum_, max_, min_, mean = arr.sum(), arr.max(), arr.min(), arr.mean()
        std = arr.std()
        print('{sum:10.3E}  {max:10.3E}  {min:10.3E}  {mean:10.3E} {std:10.3E}'
              .format(sum=sum_, max=max_, min=min_, mean=mean, std=std))
    print('')


def _is_moment_different(data, mean, std, threshold):
    mean_diff = abs(mean - np.mean(data)) / (mean or 1.0)
    std_diff = abs(std - np.std(data)) / (std or 1.0)
    print('mean diff: {} [%]'.format(100 * mean_diff))
    print('std diff: {} [%]'.format(100 * std_diff))
    return (mean_diff > threshold) or (std_diff > threshold)


def _check_dist(value, mean, std, threshold):
    _print_stats(value)
    print('Given mean: {}'.format(mean))
    print('Given stddev: {}'.format(std))
    print('Checking (threshold: {} [%])'.format(100 * threshold))
    if _is_moment_different(value, mean, std, threshold):
        raise ValueError('Data are different')
    print('Okay')


def _save_output(filepath, data, key):
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)

    print('Saving output value to {}'.format(filepath))
    print('  Shape {}'.format(data.shape))
    print('  Dtype {}'.format(data.dtype))
    file_ = h5py.File(filepath, 'w')
    file_.create_dataset(key, data=data)
    file_.close()


def main():
    args = _parse_command_line_args()
    cfg = luchador.util.load_config(args.config)
    initializer = _create_initializer(**cfg['initializer'])
    value = _run_initializer(initializer, **cfg['test_config'])

    if args.output:
        _save_output(args.output, value, args.key)

    _check_dist(value, **cfg['compare_config'])


if __name__ == '__main__':
    main()
