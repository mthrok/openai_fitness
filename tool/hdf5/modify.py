"""Modify dataset in HDF5 file"""
from __future__ import print_function
from __future__ import absolute_import

from collections import OrderedDict

import numpy as np
from .common import load_hdf5, unnest_hdf5


def _get_dataset_summary(file_):
    return OrderedDict(
        [(key, {
            'dtype': value.dtype,
            'shape': value.shape,
            'mean': np.mean(value),
            'sum': np.sum(value),
            'max': np.max(value),
            'min': np.min(value),
        }) for key, value in file_.items()])


def _max_str(list_):
    return max([len(str(e)) for e in list_])


def _print_summary(summary):
    dtype_len = _max_str([s['dtype'] for s in summary.values()]) + 1
    shape_len = _max_str([s['shape'] for s in summary.values()]) + 1
    path_len = _max_str(summary.keys()) + 1
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
    for path, smr in summary.items():
        print (
            '{path:{path_len}}{dtype:{dtype_len}}{shape:{shape_len}} '
            '{sum:10.3E}  {max:10.3E}  {min:10.3E}  {mean:10.3E}'
            .format(
                dtype=smr['dtype'], dtype_len=dtype_len,
                shape=smr['shape'], shape_len=shape_len,
                path=path, path_len=path_len, sum=smr['sum'], max=smr['max'],
                min=smr['min'], mean=smr['mean'],
            )
        )


def inspect(args):
    """Print statistics of HDF5 file"""
    file_ = unnest_hdf5(load_hdf5(args.input_file))
    _print_summary(_get_dataset_summary(file_))


def rename(args):
    """Rename dataset in HDF5 file"""
    file_ = load_hdf5(args.input_file, 'r+')
    if args.old_key not in file_:
        raise KeyError('Dataset not found: {}'.format(args.old_key))

    if args.new_key in file_:
        raise KeyError('Dataset exists: {}'.format(args.new_key))

    print('{}Renaming {} to {}'.format(
        '(dryrun) ' if args.dry_run else '', args.old_key, args.new_key))
    if not args.dry_run:
        file_[args.new_key] = file_[args.old_key]
        del file_[args.old_key]


def delete(args):
    """Delete dataset in HDF5"""
    file_ = load_hdf5(args.input_file, 'r+')
    for key in args.keys:
        if key not in file_:
            raise KeyError('Databset not found: {}'.format(key))

    for key in args.keys:
        print('{}Deleting key: {}'
              .format('(dryrun) ' if args.dry_run else '', key))
        if not args.dry_run:
            del file_[key]
