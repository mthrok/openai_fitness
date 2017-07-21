"""Dataset loading utilities"""
from __future__ import division
from __future__ import absolute_import

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
        if self.label:
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
        label = self.label[start:end, ...] if self.label else None
        return Batch(self.data[start:end, ...], label)


def load_mnist(filepath, flatten=None, data_format=None):
    """Load U of Montreal MNIST data
    http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz

    Parameters
    ----------
    filepath : str
        Path to `mnist.pkl.gz` data

    flatten : Boolean
        If True each image is flattened to 1D vector.

    Returns
    -------
    Datasets
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
        _LG.info('    Shape: %s', data.shape)
        _LG.info('    DType: %s', data.dtype)
        _LG.info('    Mean: %s', data.mean())
        _LG.info('    Max:  %s', data.max())
        _LG.info('    Min:  %s', data.min())
    return Datasets(
        Dataset(*datasets[0]), Dataset(*datasets[1]), Dataset(*datasets[2])
    )


def _prod(numbers):
    ret = 1
    for num in numbers:
        ret *= num
    return ret


def load_celeba_face(filepath, flatten=False, data_format=None):
    """Load preprocessed CelebA dataset

    To prepare dataset, follow the steps.
    1. Download aligned & cropped face images from CelebA project.
    http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
    2. Use the following to preprocess the images and pickle them.
    https://s3.amazonaws.com/luchador/dataset/celeba/create_celeba_face_dataset.py
    3. Provide the resulting filepath to this function.

    Parameters
    ----------
    filepath : str
        Path to the pickled CelebA dataset.

    flatten : Boolean
        If True each image is flattened to 1D vector.

    data_format : str
        Either 'NCHW' or 'NHWC'.

    Returns
    -------
    Datasets
    """
    _LG.info('Loading %s', filepath)
    with gzip.open(filepath, 'rb') as file_:
        datasets = pickle.load(file_)
    shape = datasets['train'].shape
    if flatten:
        datasets = {
            key: data.reshape(shape[0], -1)
            for key, data in datasets.items()
        }
    elif data_format == 'NCHW':
        datasets = {
            key: data.transpose(0, 3, 1, 2)
            for key, data in datasets.items()
        }

    datasets = {
        key: data.astype(np.float32) / 255
        for key, data in datasets.items()
    }

    for key, data in datasets.items():
        _LG.info('  %s Data Statistics', key)
        _LG.info('    Shape: %s', data.shape)
        _LG.info('    DType: %s', data.dtype)
        _LG.info('    Mean: %s', data.mean())
        _LG.info('    Max:  %s', data.max())
        _LG.info('    Min:  %s', data.min())
    return Datasets(
        Dataset(datasets['train'], None),
        Dataset(datasets['test'], None),
        Dataset(datasets['valid'], None),
    )
