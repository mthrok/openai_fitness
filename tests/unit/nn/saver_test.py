from __future__ import absolute_import

import unittest

import os
import h5py
import numpy as np

from luchador.nn.saver import Saver

OUTPUT_DIR = 'tmp'


def list_files(prefix, dir_name=OUTPUT_DIR):
    return [os.path.join(dir_name, f) for f in os.listdir(dir_name)
            if f.startswith(prefix)]


def remove_files(prefix, dir_name=OUTPUT_DIR):
    for f in list_files(prefix, dir_name):
        os.remove(f)


def gen_random_value():
    return np.random.randint(low=0, high=2, size=(32, 8, 8, 4))


class SaverTest(unittest.TestCase):
    def test_save(self):
        """`save` function can save data"""
        prefix = 'test_save'
        remove_files(prefix)
        saver = Saver(OUTPUT_DIR, prefix=prefix)

        key = 'test'
        value1 = np.random.randint(low=0, high=2, size=(32, 8, 8, 4))
        filepath = saver.save({key: value1}, global_step=0)

        f = h5py.File(filepath)
        self.assertTrue(
            np.all(value1 == f[key]),
            'Saved value do not match. '
            'Expected: {}. Found: {}'.format(value1, f[key])
        )

    def test_append(self):
        """`save` function appends to existing file"""
        prefix, global_step = 'test_append', 0
        remove_files(prefix)
        saver = Saver(OUTPUT_DIR, prefix=prefix)

        key1 = 'key1'
        value1 = gen_random_value()
        filepath1 = saver.save({key1: value1}, global_step=global_step)

        key2 = 'key2'
        value2 = gen_random_value()
        filepath2 = saver.save({key2: value2}, global_step=global_step)

        self.assertEqual(
            filepath1, filepath2, 'Data must be written to the same file.'
        )

        f = h5py.File(filepath1)
        self.assertTrue(
            np.all(value1 == f[key1]),
            'Saved values do not match. '
            'Expected: {}. Found: {}'.format(value1, value2)
        )
        self.assertTrue(
            np.all(value2 == f[key2]),
            'Saved values do not match. '
            'Expected: {}. Found: {}'.format(value1, value2)
        )

    def test_overwrite(self):
        """`save` function raises error if key exists"""
        prefix, global_step = 'test_overwrite', 0
        remove_files(prefix)
        saver = Saver(OUTPUT_DIR, prefix=prefix)

        key = 'foo'
        value1 = gen_random_value()
        value2 = value1 + 1

        filepath1 = saver.save({key: value1}, global_step=global_step)
        filepath2 = saver.save({key: value2}, global_step=global_step)

        self.assertEqual(
            filepath1, filepath2, 'Data must be written to the same file.'
        )
        f = h5py.File(filepath1)
        self.assertTrue(
            np.all(value2 == f[key]),
            'Saved value do not match. '
            'Expected: {}. Found: {}'.format(value1, f[key])
        )

    def test_max_to_keep(self):
        """Saver only keeps lates files"""
        prefix, max_to_keep = 'test_max_to_keep', 10
        remove_files(prefix)
        saver = Saver(OUTPUT_DIR, prefix=prefix, max_to_keep=max_to_keep)

        for step in range(2 * max_to_keep):
            saver.save({'foo': gen_random_value()}, global_step=step)
            files_kept = len(list_files(prefix))
            self.assertLessEqual(files_kept, max_to_keep)

    def test_keep_every_n_hours(self):
        prefix, max_to_keep = 'test_keep_every_n_hours', 10
        keep_every_n_hours = 1.0
        remove_files(prefix)
        saver = Saver(OUTPUT_DIR, prefix=prefix, max_to_keep=max_to_keep)

        for step in range(max_to_keep + 1):
            saver.save({'foo': gen_random_value()}, global_step=step)

        for i in range(10):
            saver.last_saved -= 3600 * keep_every_n_hours
            for step in range(max_to_keep * (i+1), max_to_keep * (i + 2)):
                saver.save({'foo': gen_random_value()}, global_step=step+1)
                files_kept = len(list_files(prefix))
                self.assertEqual(files_kept, max_to_keep + i + 1)
