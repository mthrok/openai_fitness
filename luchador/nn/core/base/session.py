from __future__ import absolute_import

from collections import OrderedDict

import h5py
import numpy as np

__all__ = ['Session']


def _parse_dataset(h5group, prefix=''):
    ret = {}
    for key, value in h5group.items():
        path = '{}/{}'.format(prefix, key) if prefix else key
        if isinstance(value, h5py.Group):
            ret.update(_parse_dataset(value, prefix=path))
        else:
            ret[path] = np.asarray(value)
    return ret


class Session(object):
    """Defines common interface for computation handler

    Each backend must implement the following methods:
      - __init__: Construct instance and initialize session
      - run: Run computation
      - initialize: Initialize Variable-s
    """
    def __init__(self):
        raise NotImplementedError(
            '{}.{} is not implemented yet.'
            .format(type(self).__module__, type(self).__name__)
        )

    def run(self, name, inputs, outputs, updates, givens):
        raise NotImplementedError(
            '`run` method is not yet impolemented for {}.{}.'
            .format(type(self).__module__, type(self).__name__)
        )

    def initialize(self):
        raise NotImplementedError(
            '`initialize` method is not yet impolemented for {}.{}.'
            .format(type(self).__module__, type(self).__name__)
        )

    ###########################################################################
    def load_from_file(self, filepath, var_names=None, cast=True):
        """Load variable values from HDF5 file.

        Args:
          filepath (str): File path
          var_names (None or list of str): List of variable names to retrieve
            from file. If None, it tries to retrieve and assign all variables
            in the file.
          cast (Bool): If True, cast dtype automatically.
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

        self.load_dataset(data_set, cast=cast)

    def load_dataset(self, dataset, cast=True):
        raise NotImplementedError(
            '`load_dataset` method is not yet impolemented for {}.{}.'
            .format(type(self).__module__, type(self).__name__)
        )
