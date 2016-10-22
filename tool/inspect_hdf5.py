#!/usr/bin/env python

"""Command line tool to edit HDF5 file"""

from __future__ import print_function

import sys
from collections import OrderedDict
from argparse import ArgumentParser as AP

import h5py
import numpy as np


def load_hdf5(filepath, mode='r'):
    """Load HDF5 file and unnest structure"""
    return h5py.File(filepath, mode)


def unnest_hdf5(obj, prefix='', ret=None):
    if ret is None:
        ret = OrderedDict()

    for key, value in obj.items():
        path = '{}/{}'.format(prefix, key)
        if isinstance(value, h5py.Group):
            unnest_hdf5(value, path, ret)
        else:
            ret[path] = value
    return ret


def get_dataset_summary(f):
    return OrderedDict(
        [(key, {
            'dtype': value.dtype,
            'shape': value.shape,
            'mean': np.mean(value),
            'sum': np.sum(value),
            'max': np.max(value),
            'min': np.min(value),
        }) for key, value in f.items()])


def max_str(l):
    return max(map(lambda e: len(str(e)), l))


def print_summary(summary):
    dtype_len = max_str([s['dtype'] for s in summary.values()]) + 1
    shape_len = max_str([s['shape'] for s in summary.values()]) + 1
    path_len = max_str(summary.keys()) + 1
    print (
        '{path:{path_len}}{dtype:{dtype_len}}{shape:{shape_len}} '
        '{sum:>10}  {max:>10}  {min:>10}  {mean:>10}'
        .format(
            dtype='dtype', dtype_len=dtype_len,
            shape='shape', shape_len=shape_len,
            path='path', path_len=path_len,
            sum='sum', max='max', min='min', mean='mean'
        )
    )
    for path, s in summary.items():
        print (
            '{path:{path_len}}{dtype:{dtype_len}}{shape:{shape_len}} '
            '{sum:10.3E}  {max:10.3E}  {min:10.3E}  {mean:10.3E}'
            .format(
                dtype=s['dtype'], dtype_len=dtype_len,
                shape=s['shape'], shape_len=shape_len,
                path=path, path_len=path_len,
                sum=s['sum'], max=s['max'], min=s['min'], mean=s['mean'],
            )
        )


class HDF5Editor(object):
    def __init__(self):
        ap = AP(
            description='Inspect HDF5 Data'
        )
        ap.add_argument('command')

        valid_commands = ['inspect', 'delete', 'rename', 'view']
        args = ap.parse_args(sys.argv[1:2])
        if args.command not in valid_commands:
            command = 'inspect'
            argv = sys.argv[1:]
        else:
            command = args.command
            argv = sys.argv[2:]

        getattr(self, command)(argv)

    def inspect(self, argv):
        ap = AP(
            description='List up datasets in the given file.',
            usage='{} {} [-h] input_file'.format(sys.argv[0], 'inspect')
        )
        ap.add_argument('input_file', help='Input HDF5 file')
        args = ap.parse_args(argv)

        f = unnest_hdf5(load_hdf5(args.input_file))
        print_summary(get_dataset_summary(f))

    def delete(self, argv):
        ap = AP(
            description='Delete a dataset from H5 file',
            usage=('{} {} [-h] input_file keys [keys ...]'
                   .format(sys.argv[0], 'delete'))
        )
        ap.add_argument('input_file', help='Input HDF5 file.')
        ap.add_argument('keys', nargs='+', help='Names of dataset to delete')
        ap.add_argument(
            '--dry-run', '--dryrun', action='store_true',
            help='Do not apply change to the file.'
        )
        args = ap.parse_args(argv)

        f = load_hdf5(args.input_file, 'r+')
        for key in args.keys:
            if key not in f:
                raise KeyError('Databset not found: {}'.format(key))

        for key in args.keys:
            print('{}Deleting key: {}'
                  .format('(dryrun) ' if args.dry_run else '', key))
            if not args.dry_run:
                del f[key]

    def rename(self, argv):
        ap = AP(
            description='Rename a dataset in H5 file',
            usage=('{} {} [-h] input_file old_key new_key'
                   .format(sys.argv[0], 'rename'))
        )
        ap.add_argument('input_file', help='Input H5 file.')
        ap.add_argument('old_key', help='Dataset to rename')
        ap.add_argument('new_key', help='New Dataset name')
        ap.add_argument(
            '--force', '-f',
            help='Overwrite in case the dataset with new_key exists.'
        )
        ap.add_argument(
            '--dry-run', '--dryrun', action='store_true',
            help='Do not apply change to the file.'
        )
        args = ap.parse_args(argv)

        f = load_hdf5(args.input_file, 'r+')
        if args.old_key not in f:
            raise KeyError('Dataset not found: {}'.format(args.old_key))

        if args.new_key in f:
            raise KeyError('Dataset exists: {}'.format(args.new_key))

        print('{}Renaming {} to {}'.format(
            '(dryrun) ' if args.dry_run else '', args.old_key, args.new_key))
        if not args.dry_run:
            f[args.new_key] = f[args.old_key]
            del f[args.old_key]

    def view(self, argv):
        import matplotlib.pyplot as plt

        ap = AP(
            description='Visualize tensors',
        )
        ap.add_argument('input_file', help='Input H5 file.')
        ap.add_argument('key', help='Datasets to visualize')
        ap.add_argument(
            '--batch', type=int, default=0,
            help='Batch number to visualize if dataset is 4D'
        )
        ap.add_argument(
            '--format', default='NCHW',
            help='Data format. Either NCHW or NHWC. Default: NCHW'
        )
        ap.add_argument(
            '--vmax', type=float,
            help='Maximum luminance cut-off value'
        )
        ap.add_argument(
            '--vmin', type=float,
            help='Minimum luminance cut-off value'
        )
        args = ap.parse_args(argv)

        f = load_hdf5(args.input_file, 'r')
        data = np.asarray(f[args.key])

        if data.ndim == 4 and args.format == 'NHWC':
            data.transpose((0, 3, 1, 2))

        if data.ndim == 2:
            batch = 0
            data = data[None, None, :, :]
        else:
            batch = args.batch

        n_filters = data.shape[1]
        n_rows = np.floor(np.sqrt(n_filters))
        n_cols = np.ceil(n_filters / n_rows)

        vmin = args.vmin if args.vmin else data.min()
        vmax = args.vmax if args.vmax else data.max()
        fig = plt.figure()
        fig.suptitle('{}\nBatch: {}'.format(args.input_file, batch))
        for index, filter_ in enumerate(data[batch], start=1):
            axis = fig.add_subplot(n_rows, n_cols, index)
            img = axis.imshow(filter_, vmin=vmin, vmax=vmax,
                              cmap='Greys', interpolation='nearest')
            axis.set_title('Filter: {}'.format(index))
            if index == 1:
                fig.colorbar(img, ax=axis)
        print('Plot ready')
        plt.show()


if __name__ == '__main__':
    HDF5Editor()
