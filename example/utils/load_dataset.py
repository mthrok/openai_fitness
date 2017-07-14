"""Dataset loading utilities"""
import gzip
import pickle
import logging
from collections import namedtuple

import numpy as np

_LG = logging.getLogger(__name__)


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


def load_mnist(filepath, flatten):
    """Load U of Montreal MNIST data
    http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz

    Parameters
    ----------
    filepath : str
        Path to `mnist.pkl.gz` data

    flatten : Boolean
        If True each image is flattened to 1D vector.
    """
    _LG.info('Loading data %s', filepath)
    with gzip.open(filepath, 'rb') as file_:
        datasets = pickle.load(file_)
    if flatten:
        datasets = [(data.reshape(-1, 784), label) for data, label in datasets]
    return Datasets(
        Dataset(*datasets[0]), Dataset(*datasets[1]), Dataset(*datasets[2])
    )
