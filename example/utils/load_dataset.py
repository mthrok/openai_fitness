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
    'Datasets', field_names=('train', 'valid', 'test')
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
        if self.label is not None:
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
        label = None if self.label is None else self.label[start:end, ...]
        return Batch(self.data[start:end, ...], label)


def _format_dataset(datasets, flatten, data_format):
    if flatten:
        datasets = [
            (data.reshape(data.shape[0], -1), label)
            for data, label in datasets
        ]
    elif data_format == 'NHWC':
        datasets = [
            (data.transpose(0, 2, 3, 1), label)
            for data, label in datasets
        ]
    for key, (data, _) in zip(['Train', 'Valid', 'Test'], datasets):
        _LG.info('  %s Data Statistics', key)
        _LG.info('    Shape: %s', data.shape)
        _LG.info('    DType: %s', data.dtype)
        _LG.info('    Mean: %s', data.mean())
        _LG.info('    Max:  %s', data.max())
        _LG.info('    Min:  %s', data.min())
    return Datasets(
        Dataset(*datasets[0]), Dataset(*datasets[1]), Dataset(*datasets[2])
    )


def _load_mnist(filepath, mock):
    if mock:
        _LG.info('Creating mock data')
        return [
            (
                np.random.uniform(size=[n, 1, 28, 28]).astype(np.float32),
                np.random.randint(10, size=[n]),
            )
            for n in [1000, 100, 100]
        ]

    _LG.info('Loading %s', filepath)
    with gzip.open(filepath, 'rb') as file_:
        return [
            (data.reshape(-1, 1, 28, 28), label)
            for data, label in pickle.load(file_)
        ]


def load_mnist(filepath, flatten=None, data_format=None, mock=False):
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
    datasets = _load_mnist(filepath, mock)
    return _format_dataset(datasets, flatten, data_format)


def _load_celeba_face(filepath, mock):
    if mock:
        _LG.info('Creating mock data')
        return [
            (
                np.random.uniform(size=[n, 3, 64, 64]).astype(np.float32),
                None
            )
            for n in [1000, 100, 100]
        ]
    _LG.info('Loading %s', filepath)
    with gzip.open(filepath, 'rb') as file_:
        return [
            (data.astype(np.float32) / 255, label)
            for data, label in pickle.load(file_)
        ]


def load_celeba_face(filepath, flatten=False, data_format=None, mock=False):
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
    datasets = _load_celeba_face(filepath, mock)
    return _format_dataset(datasets, flatten, data_format)
