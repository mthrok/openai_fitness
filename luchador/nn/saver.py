from __future__ import division
from __future__ import absolute_import

import os
import time
import errno
import logging

from collections import OrderedDict

import h5py

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

        self.saved = OrderedDict()
        self.last_saved = time.time()

    def _write(self, f, data):
        for key, value in data.items():
            _LG.debug('  Saving: {:10} {:24} {}'.format(
                value.dtype, value.shape, key))
            if key in f:
                del f[key]
            f.create_dataset(key, data=value, chunks=True)
        f.flush()

    def save(self, data, global_step):
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

        self.saved[filepath] = time.time()
        self._remove_old_data()
        return filepath

    def _remove_old_data(self):
        threshold = 3600 * self.keep_every_n_hours

        while self.max_to_keep < len(self.saved):
            filepath, saved_time = self.saved.popitem(last=False)
            elapsed = (saved_time - self.last_saved)
            if elapsed > threshold:
                self.last_saved = saved_time
            else:
                try:
                    _LG.info('Removing: {}'.format(filepath))
                    os.remove(filepath)
                except Exception:
                    _LG.exception('Failed to delete {}'.format(filepath))
