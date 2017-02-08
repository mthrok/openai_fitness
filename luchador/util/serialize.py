"""Define common method for [de]serializing NumPyArray"""
from __future__ import absolute_import

import zlib
import numpy as np

__all__ = ['serialize_numpy_array', 'deserialize_numpy_array']


def serialize_numpy_array(array, compress=True):
    """Serialize NumPy ND Array into JSON format

    Parameters
    ----------
    array : NumPy ND Array
        Array to serialize
    compress : bool
        When True, array data are compressed. Default: True.

    Returns
    -------
    dict
        Serialized array

        shape : list
            array shape
        dtype : str
            dtype
        data : byte string
            actual data
        compressed : bool
            Indicates if data is compressed
    """
    str_ = array.tostring()
    if compress:
        str_ = zlib.compress(str_)
    return {
        'shape': array.shape,
        'dtype': array.dtype.name,
        'data': str_.encode('base64'),
        'compressed': compress,
    }


def deserialize_numpy_array(obj):
    """Deserialize NumPy ND Array from JSON format

    Parameters
    ----------
    obj : dict
        See `serialize_numpy_array` for the detail.

    Returns
    -------
    NumPy ND Array
        Resulting array
    """
    data = obj['data'].decode('base64')
    if obj['compressed']:
        data = zlib.decompress(data)
    array = np.fromstring(data, dtype=obj['dtype'])
    return array.reshape(obj['shape'])
