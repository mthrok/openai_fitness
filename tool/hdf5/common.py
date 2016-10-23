"""Common functionality used by HDF5 inspection submodules"""

from __future__ import absolute_import

from collections import OrderedDict

import h5py


def load_hdf5(filepath, mode='r'):
    """Load HDF5 file and unnest structure"""
    return h5py.File(filepath, mode)


def unnest_hdf5(obj, prefix='', ret=None):
    """Turn recursive structure of HDF5 into flat path-dataset mapping"""
    if ret is None:
        ret = OrderedDict()

    for key, value in obj.items():
        path = '{}/{}'.format(prefix, key)
        if isinstance(value, h5py.Group):
            unnest_hdf5(value, path, ret)
        else:
            ret[path] = value
    return ret
