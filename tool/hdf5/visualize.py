"""Visualize dataset in HDF5 file"""
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

from .common import load_hdf5


def visualize_dataset(args):
    """Visualize 2D/4D data in HDF5 file"""
    import matplotlib.pyplot as plt

    file_ = load_hdf5(args.input_file, 'r')
    data = np.asarray(file_[args.key])

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
                          cmap='jet', interpolation='nearest')
        axis.set_title('Filter: {}'.format(index))
        if index == 1:
            fig.colorbar(img, ax=axis)
    print('Plot ready')
    plt.show()
