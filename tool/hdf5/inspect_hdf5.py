#!/usr/bin/env python
"""Command line tool to edit HDF5 file"""
from __future__ import print_function
from __future__ import absolute_import

import argparse

from util import modify, visualize


def _add_inspect_command(subparsers):
    parser = subparsers.add_parser(
        'inspect',
        description='List up datasets in the given file.',
    )
    parser.add_argument('input_file', help='Input HDF5 file')
    parser.set_defaults(func=modify.inspect)


def _add_delete_command(subparsers):
    parser = subparsers.add_parser(
        'delete',
        description='Delete a dataset from H5 file',
    )
    parser.add_argument('input_file', help='Input HDF5 file.')
    parser.add_argument('keys', nargs='+', help='Names of dataset to delete')
    parser.add_argument(
        '--dry-run', '--dryrun', action='store_true',
        help='Do not apply change to the file.'
    )
    parser.set_defaults(func=modify.delete)


def _add_rename_command(subparsers):
    parser = subparsers.add_parser(
        'rename',
        description='Rename a dataset in H5 file',
    )
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
    parser.set_defaults(func=modify.rename)


def _add_view_command(subparsers):
    parser = subparsers.add_parser(
        'view',
        description='Visualize tensors',
    )
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
    parser.set_defaults(func=visualize.visualize_dataset)


def _main():
    parser = argparse.ArgumentParser(
        description='Inspect HDF5 Data'
    )
    subparsers = parser.add_subparsers()
    _add_inspect_command(subparsers)
    _add_delete_command(subparsers)
    _add_rename_command(subparsers)
    _add_view_command(subparsers)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    _main()
