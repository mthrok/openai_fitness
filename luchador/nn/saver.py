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


class Saver(object):
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

    def _write(self, f, data):
        for key, value in data.items():
            _LG.debug('  Saving: {:10} {:24} {}'.format(
                value.dtype, value.shape, key))
            if key in f:
                del f[key]

            chunks = False if value.size == 1 else True
            f.create_dataset(key, data=value, chunks=chunks)

        if 'LUCHADOR_NN_BACKEND' not in f:
            data = np.string_(luchador.get_nn_backend())
            f.create_dataset('LUCHADOR_NN_BACKEND', data=data, dtype='S10')
        if 'LUCHADOR_NN_CONV_FORMAT' not in f:
            data = np.string_(luchador.get_nn_conv_format())
            f.create_dataset('LUCHADOR_NN_CONV_FORMAT', data=data, dtype='S4')
        if 'LUCHADOR_NN_DTYPE' not in f:
            data = np.string_(luchador.get_nn_dtype())
            f.create_dataset('LUCHADOR_NN_DTYPE', data=data, dtype='S10')
        if 'LUCHADOR_VERSION' not in f:
            data = np.string_(luchador.__version__)
            f.create_dataset('LUCHADOR_VERSION', data=data)
        f.flush()

    def save(self, data, global_step, now=None):
        # `now` should be used only for testing
        filename = '{}_{}.h5'.format(self.prefix, global_step)
        filepath = os.path.join(self.output_dir, filename)

        _LG.info('Saving data to {}'.format(filepath))
        f = h5py.File(filepath, 'a')
        try:
            self._write(f, data)
        except Exception:
            raise
        finally:
            f.close()

        self._add_new_record(filepath, now=now)
        self._remove_old_data()
        return filepath

    def _add_new_record(self, filepath, now=None):
        # `now` should be used only for testing
        now = now if now else time.time()
        if self.last_back_up is None:
            # If this is the first time, do not put this in delete candidates
            _LG.info('Backing up: {}'.format(filepath))
            self.last_back_up = now
        elif now - self.last_back_up < self.threshold:
            # Add the new file to delete candidate
            self.to_be_deleted.append((now, filepath))
        else:
            # It's been a while since the last back up, so back up one
            if self.to_be_deleted:
                self.last_back_up, filepath_ = self.to_be_deleted.pop()
                self.to_be_deleted.append((now, filepath))
                _LG.info('Backing up: {}'.format(filepath_))
            else:
                self.last_back_up = now

    def _remove_old_data(self):
        while self.max_to_keep < len(self.to_be_deleted):
            _, filepath = self.to_be_deleted.pop(0)
            try:
                _LG.info('Removing: {}'.format(filepath))
                os.remove(filepath)
            except Exception:
                _LG.exception('Failed to delete {}'.format(filepath))
