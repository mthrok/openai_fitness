"""Dataset loading utilities"""
import gzip
import pickle
import logging
from collections import namedtuple

import numpy as np

_LG = logging.getLogger(__name__)
# pylint:disable=invalid-name


Datasets = namedtuple(
    'Datasets', field_names=('train', 'test', 'validation')
)

Batch = namedtuple(
    'Batch', field_names=('data', 'label')
)


class Dataset(object):
    """Dataset with simple mini batch mechanism"""
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.n_data = len(data)
        self.index = 0

    @property
    def shape(self):
        """Get undixed batch shape"""
        return (None,) + self.data.shape[1:]

    def _shuffle(self):
        perm = np.arange(self.n_data)
        np.random.shuffle(perm)
        self.data = self.data[perm]
        self.label = self.label[perm]

    def next_batch(self, batch_size):
        """Get mini batch.

        Parameters
        ----------
        batch_size : int
            The number of data point to fetch

        Returns
        -------
        Batch
            `data` and `label` attributes.
        """
        if self.index + batch_size > self.n_data:
            self._shuffle()
            self.index = 0
        start, end = self.index, self.index + batch_size
        self.index += batch_size
        return Batch(self.data[start:end, ...], self.label[start:end, ...])


def load_mnist(filepath, flatten=None, data_format=None):
    """Load U of Montreal MNIST data
    http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz

    Parameters
    ----------
    filepath : str
        Path to `mnist.pkl.gz` data

    flatten : Boolean
        If True each image is flattened to 1D vector.
    """
    _LG.info('Loading %s', filepath)
    with gzip.open(filepath, 'rb') as file_:
        datasets = pickle.load(file_)
    reshape = None
    if flatten:
        reshape = (-1, 784)
    elif data_format == 'NCHW':
        reshape = (-1, 1, 28, 28)
    elif data_format == 'NHWC':
        reshape = (-1, 28, 28, 1)

    if reshape:
        datasets = [(data.reshape(*reshape), lbl) for data, lbl in datasets]

    for (data, _), key in zip(datasets, ['Train', 'Test', 'Validation']):
        _LG.info('  %s Data Statistics', key)
        _LG.info('    Dtype: %s', data.dtype)
        _LG.info('    Mean: %s', data.mean())
        _LG.info('    Max:  %s', data.max())
        _LG.info('    Min:  %s', data.min())
    return Datasets(
        Dataset(*datasets[0]), Dataset(*datasets[1]), Dataset(*datasets[2])
    )
