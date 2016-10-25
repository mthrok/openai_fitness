"""Implements Session functionalities common to backends"""

from __future__ import absolute_import

import abc
import logging
from collections import OrderedDict

import h5py
import numpy as np

_LG = logging.getLogger(__name__)


def _parse_dataset(h5group, prefix=''):
    meta_data = ['LUCHADOR_VERSION', 'LUCHADOR_NN_BACKEND',
                 'LUCHADOR_NN_CONV_FORMAT', 'LUCHADOR_NN_DTYPE']

    ret = OrderedDict()
    for key, value in h5group.items():
        if key in meta_data:
            _LG.info('  %-25s %s', key, np.asarray(value))
            continue

        path = '{}/{}'.format(prefix, key) if prefix else key
        if isinstance(value, h5py.Group):
            ret.update(_parse_dataset(value, prefix=path))
        else:
            ret[path] = np.asarray(value)
    return ret


class BaseSession(object):
    """Defines common interface for computation handler

    Each backend must implement the following methods:
      - __init__: Construct instance and initialize session
      - run: Run computation
      - initialize: Initialize Variable-s
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass

    @abc.abstractmethod
    def run(self, inputs=None, outputs=None,
            updates=None, givens=None, name=None):
        """Runs operations and evaluates tensors in outputs"""
        pass

    @abc.abstractmethod
    def initialize(self):
        """Initialize Variables"""
        pass

    @abc.abstractproperty
    def graph(self):
        """Mimic TF.Session.graph for SummaryWriter"""
        return None

    ###########################################################################
    def load_from_file(self, filepath, var_names=None, cast=True, strict=True):
        """Load variable values from HDF5 file.

        Args:
          filepath (str): File path

          var_names (None or list of str): List of variable names to retrieve
            from file. If None, it tries to retrieve and assign all variables
            in the file.

          cast (Bool): If True, cast dtype automatically.

          strict (Bool): When True, if dataset contains a value for Variable
            which is not defined, then ValueError exception is raised.
            Otherwise it will be skipped.
        """
        f = h5py.File(filepath, 'r')
        try:
            data_set = _parse_dataset(f)
        except Exception:
            raise
        finally:
            f.close()

        if var_names is not None:
            data_set = OrderedDict([(n, data_set[n]) for n in var_names])

        self.load_dataset(data_set, cast=cast, strict=strict)

    @abc.abstractmethod
    def load_dataset(self, dataset, cast=True, strict=True):
        """Set the values of Variables with the given dataset values

        TODO: Add args
        """
        pass
