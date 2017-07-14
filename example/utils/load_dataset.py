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
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.n_data = len(data)
        self.index = 0

    def _shuffle(self):
        perm = np.arange(self.n_data)
        np.random.shuffle(perm)
        self.data = self.data[perm]
        self.label = self.label[perm]

    def next_batch(self, batch_size):
        if self.index + batch_size > self.n_data:
            self._shuffle()
            self.index = 0
        start, end = self.index, self.index + batch_size
        self.index += batch_size
        return Batch(self.data[start:end], self.label[start:end])


def load_mnist(filepath, flatten):
    _LG.info('Loading data %s', filepath)
    with gzip.open(filepath, 'rb') as file_:
        datasets = pickle.load(file_)
    if flatten:
        datasets = [
            (data.reshape(-1, 784), label)
            for data, label in datasets
        ]
    return Datasets(
        Dataset(*datasets[0]), Dataset(*datasets[1]), Dataset(*datasets[2])
    )
