"""Implements parameter saver"""
from __future__ import division
from __future__ import absolute_import

import os
import time
import errno
import logging

import h5py
import numpy as np

import luchador

_LG = logging.getLogger(__name__)

__all__ = ['Saver']


def _write_data_to_file(file_, data):
    for key, value in data.items():
        _LG.debug('  Saving: %10s %24s %s', value.dtype, value.shape, key)
        if key in file_:
            del file_[key]

        chunks = None if value.size == 1 else True
        file_.create_dataset(key, data=value, chunks=chunks)

    if 'LUCHADOR_NN_BACKEND' not in file_:
        data = np.string_(luchador.get_nn_backend())
        file_.create_dataset('LUCHADOR_NN_BACKEND', data=data, dtype='S10')
    if 'LUCHADOR_NN_CONV_FORMAT' not in file_:
        data = np.string_(luchador.get_nn_conv_format())
        file_.create_dataset('LUCHADOR_NN_CONV_FORMAT', data=data, dtype='S4')
    if 'LUCHADOR_NN_DTYPE' not in file_:
        data = np.string_(luchador.get_nn_dtype())
        file_.create_dataset('LUCHADOR_NN_DTYPE', data=data, dtype='S10')
    if 'LUCHADOR_VERSION' not in file_:
        data = np.string_(luchador.__version__)
        file_.create_dataset('LUCHADOR_VERSION', data=data)
    file_.flush()


class Saver(object):
    """Save set of NumPy arrays into HDF5 format

    Works in the way similar to tensorflow Saver

    Parameters
    ----------
    output_dir : str
        Directory to save the resulting data

    max_to_keep : int
        The number of the latest save data to keep

    keep_every_n_hours : float
        Keep file every this hour

    prefix : str
        file name prefix
    """
    def __init__(self, output_dir, max_to_keep=5,
                 keep_every_n_hours=1.0, prefix='save'):
        try:
            os.makedirs(output_dir)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

        self.prefix = prefix
        self.output_dir = output_dir
        self.max_to_keep = max_to_keep
        self.keep_every_n_hours = keep_every_n_hours
        self.threshold = 3600 * keep_every_n_hours

        self.to_be_deleted = []
        self.last_back_up = None

    def save(self, data, global_step, now=None):
        """Save data to HDF5 file

        Parameters
        ----------
        data : dict
            keys are the name of variables
            values are NumPy data

        global_step : int
            global step at the time this data was created

        now : int
            unix time. only used for test
        """
        filename = '{}_{}.h5'.format(self.prefix, global_step)
        filepath = os.path.join(self.output_dir, filename)

        _LG.info('Saving data to %s', filepath)
        file_ = h5py.File(filepath, 'a')
        try:
            _write_data_to_file(file_, data)
        except Exception:
            raise
        finally:
            file_.close()

        self._add_new_record(filepath, now=now)
        self._remove_old_data()
        return filepath

    def _add_new_record(self, filepath, now=None):
        # `now` should be used only for testing
        now = now if now else time.time()
        if self.last_back_up is None:
            # If this is the first time, do not put this in delete candidates
            _LG.info('Backing up: %s', filepath)
            self.last_back_up = now
        elif now - self.last_back_up < self.threshold:
            # Add the new file to delete candidate
            self.to_be_deleted.append((now, filepath))
        else:
            # It's been a while since the last back up, so back up one
            if self.to_be_deleted:
                self.last_back_up, filepath_ = self.to_be_deleted.pop()
                self.to_be_deleted.append((now, filepath))
                _LG.info('Backing up: %s', filepath_)
            else:
                self.last_back_up = now

    def _remove_old_data(self):
        if not self.to_be_deleted:
            return

        to_be_deleted = []
        while True:
            then = self.to_be_deleted[0][0]
            if then - self.last_back_up > self.threshold:
                _, filepath = self.to_be_deleted.pop(0)
                to_be_deleted.append(filepath)
            else:
                break

        while self.max_to_keep < len(self.to_be_deleted):
            _, filepath = self.to_be_deleted.pop(0)
            to_be_deleted.append(filepath)

        for filepath in to_be_deleted:
            try:
                _LG.info('Removing: %s', filepath)
                os.remove(filepath)
            except OSError:
                _LG.exception('Failed to delete %s', filepath)
