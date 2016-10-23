#!/usr/bin/env python

"""Command line tool to edit HDF5 file"""

from __future__ import print_function
from __future__ import absolute_import

import sys
from argparse import ArgumentParser as AP

from hdf5.modify import inspect, rename, delete
from hdf5.visualize import visualize_dataset


def _inspect():
    parser = AP(
        description='List up datasets in the given file.',
    )
    parser.add_argument('inspect')
    parser.add_argument('input_file', help='Input HDF5 file')
    args = parser.parse_args()
    inspect(args)


def _delete():
    parser = AP(
        description='Delete a dataset from H5 file',
    )
    parser.add_argument('delete')
    parser.add_argument('input_file', help='Input HDF5 file.')
    parser.add_argument('keys', nargs='+', help='Names of dataset to delete')
    parser.add_argument(
        '--dry-run', '--dryrun', action='store_true',
        help='Do not apply change to the file.'
    )
    args = parser.parse_args()
    delete(args)


def _rename():
    parser = AP(
        description='Rename a dataset in H5 file',
    )
    parser.add_argument('rename')
    parser.add_argument('input_file', help='Input H5 file.')
    parser.add_argument('old_key', help='Dataset to rename')
    parser.add_argument('new_key', help='New Dataset name')
    parser.add_argument(
        '--force', '-f',
        help='Overwrite in case the dataset with new_key exists.'
    )
    parser.add_argument(
        '--dry-run', '--dryrun', action='store_true',
        help='Do not apply change to the file.'
    )
    args = parser.parse_args()
    rename(args)


def _visualize_dataset():
    parser = AP(
        description='Visualize tensors',
    )
    parser.add_argument('view')
    parser.add_argument('input_file', help='Input H5 file.')
    parser.add_argument('key', help='Datasets to visualize')
    parser.add_argument(
        '--batch', type=int, default=0,
        help='Batch number to visualize if dataset is 4D'
    )
    parser.add_argument(
        '--format', default='NCHW',
        help='Data format. Either NCHW or NHWC. Default: NCHW'
    )
    parser.add_argument(
        '--vmax', type=float,
        help='Maximum luminance cut-off value'
    )
    parser.add_argument(
        '--vmin', type=float,
        help='Minimum luminance cut-off value'
    )
    args = parser.parse_args()
    visualize_dataset(args)


SUBCOMMANDS = {
    'inspect': _inspect,
    'delete': _delete,
    'rename': _rename,
    'view': _visualize_dataset,
}


def _main():
    parser = AP(
        description='Inspect HDF5 Data'
    )
    parser.add_argument(
        'command',
        choices=['inspect', 'delete', 'rename', 'view']
    )

    args = parser.parse_args(sys.argv[1:2])
    SUBCOMMANDS[args.command]()

if __name__ == '__main__':
    _main()
