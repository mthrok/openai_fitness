#!/usr/bin/env python
"""Create sample batch HDF5 data from MNIST dataset"""
from __future__ import absolute_import

import os
import gzip
import cPickle as pickle

import h5py
import numpy as np


def _parse_command_line_args():
    from argparse import ArgumentParser as AP
    ap = AP(
        description=(
            'Create sample batch from MNIST dataset. '
            '(http://deeplearning.net/data/mnist/mnist.pkl.gz) '
            'Resulting batch has shape (N=10, C[=4], H[=28], W[=27])'
            'where the first index corresponds to digit.'
        )
    )
    ap.add_argument('output', help='Output file path')
    ap.add_argument('--input', help='mnist.pkl data',
                    default=os.path.join('data', 'mnist.pkl.gz'))
    ap.add_argument('--key', default='input', help='Name of dataset to create')
    ap.add_argument('--height', default=28, type=int, help='batch height.')
    ap.add_argument('--width', default=27, type=int, help='batch width.')
    ap.add_argument('--channel', default=4,
                    type=int, help='#Channels in batch.')
    ap.add_argument('--plot', action='store_true',
                    help='Visualize the resulting data')
    return ap.parse_args()


def _load_data(filepath):
    with gzip.open(filepath, 'rb') as file_:
        return pickle.load(file_)[0]


def _process(dataset, channel=4, height=28, width=27):
    batch, ret, sample = 0, [], []
    for datum, label in zip(*dataset):
        if label == batch:
            sample.append(datum.reshape((28, 28))[:height, :width])
        if len(sample) == channel:
            ret.append(sample)
            sample = []
            batch += 1
    return np.asarray(ret, dtype=np.float32)


def _plot(batch):
    import matplotlib.pyplot as plt

    n_channels = batch.shape[1]
    n_row = np.ceil(np.sqrt(n_channels))
    n_col = n_channels // n_row
    for batch, sample in enumerate(batch):
        fig = plt.figure()
        for channel, data in enumerate(sample):
            ax = fig.add_subplot(n_row, n_col, channel+1)
            ax.imshow(255 * data, cmap='Greys')
            ax.set_title('Batch: {}, Channel: {}'.format(batch, channel))
    plt.show()


def _save_data(key, data, filepath):
    file_ = h5py.File(filepath, 'a')
    if key in file_:
        del file_[key]
    file_.create_dataset(key, data=data)
    file_.close()


def _main():
    args = _parse_command_line_args()
    data = _process(
        _load_data(args.input), args.channel, args.height, args.width)
    if args.plot:
        _plot(data)
    _save_data(args.key, data, args.output)


if __name__ == '__main__':
    _main()
