from __future__ import absolute_import

import unittest

import os
import h5py
import numpy as np

from luchador.nn.saver import Saver

OUTPUT_DIR = os.path.join('tmp', 'saver_test')
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# pylint: disable=invalid-name


def _list_files(prefix, dir_name=OUTPUT_DIR):
    if not os.path.exists(dir_name):
        return []
    return [
        os.path.join(dir_name, f)
        for f in os.listdir(dir_name) if f.startswith(prefix)
    ]


def _remove_files(dir_name=OUTPUT_DIR):
    for file_ in _list_files('', dir_name):
        os.remove(file_)


def _gen_random_value():
    return np.random.randint(low=0, high=2, size=(32, 8, 8, 4))


class SaverTest(unittest.TestCase):
    def _get_empty_dir(self):
        output_dir = os.path.join(OUTPUT_DIR, self.id().split('.')[-1])
        _remove_files(output_dir)
        return output_dir

    def test_save(self):
        """`save` function can save data"""
        prefix = 'test_save'

        output_dir = self._get_empty_dir()
        saver = Saver(output_dir, prefix=prefix)

        key = 'test'
        value1 = np.random.randint(low=0, high=2, size=(32, 8, 8, 4))
        filepath = saver.save({key: value1}, global_step=0)

        file_ = h5py.File(filepath)
        self.assertTrue(
            np.all(value1 == file_[key]),
            'Saved value do not match. '
            'Expected: {}. Found: {}'.format(value1, file_[key])
        )

    def test_append(self):
        """`save` function appends to existing file"""
        prefix, global_step = 'test_append', 0

        output_dir = self._get_empty_dir()
        saver = Saver(output_dir, prefix=prefix)

        key1 = 'key1'
        value1 = _gen_random_value()
        filepath1 = saver.save({key1: value1}, global_step=global_step)

        key2 = 'key2'
        value2 = _gen_random_value()
        filepath2 = saver.save({key2: value2}, global_step=global_step)

        self.assertEqual(
            filepath1, filepath2, 'Data must be written to the same file.'
        )

        file_ = h5py.File(filepath1)
        self.assertTrue(
            np.all(value1 == file_[key1]),
            'Saved values do not match. '
            'Expected: {}. Found: {}'.format(value1, value2)
        )
        self.assertTrue(
            np.all(value2 == file_[key2]),
            'Saved values do not match. '
            'Expected: {}. Found: {}'.format(value1, value2)
        )

    def test_overwrite(self):
        """`save` function overwrites the existing values"""
        prefix, global_step = 'test_overwrite', 0

        output_dir = self._get_empty_dir()
        saver = Saver(output_dir, prefix=prefix)

        key = 'foo'
        value1 = _gen_random_value()
        value2 = value1 + 1

        filepath1 = saver.save({key: value1}, global_step=global_step)
        filepath2 = saver.save({key: value2}, global_step=global_step)

        self.assertEqual(
            filepath1, filepath2, 'Data must be written to the same file.'
        )
        file_ = h5py.File(filepath1)
        self.assertTrue(
            np.all(value2 == file_[key]),
            'Saved value do not match. '
            'Expected: {}. Found: {}'.format(value1, file_[key])
        )

    def test_max_to_keep(self):
        """Saver only keeps lates files"""
        prefix, max_to_keep = 'test_max_to_keep', 10

        output_dir = self._get_empty_dir()
        saver = Saver(output_dir, prefix=prefix, max_to_keep=max_to_keep)

        for step in range(2 * max_to_keep):
            saver.save({'foo': _gen_random_value()}, global_step=step)
            files_kept = len(_list_files(prefix, output_dir))
            self.assertLessEqual(files_kept, max_to_keep + 1)

    def test_keep_every_n_hours(self):
        prefix, max_to_keep = 'test_keep_every_n_hours', 10
        keep_every_n_hours = 1.0

        output_dir = self._get_empty_dir()
        saver = Saver(output_dir, prefix=prefix, max_to_keep=max_to_keep,
                      keep_every_n_hours=keep_every_n_hours)

        for step in range(max_to_keep + 1):
            saver.save({'foo': _gen_random_value()}, global_step=step)

        for i in range(10):
            now = 3600 * i
            for step in range(max_to_keep * (i+1), max_to_keep * (i + 2)):
                saver.save(
                    {'foo': _gen_random_value()},
                    global_step=step+1, now=now
                )
                files_kept = len(_list_files(prefix, output_dir))
                self.assertEqual(files_kept, max_to_keep+1)

    def test_keep_oldest_file_while_delete(self):
        """Saver retains oldest file in current tmp files"""
        prefix, max_to_keep = 'test_keep_and_keep', 3
        keep_every_n_hours = 0.5

        output_dir = self._get_empty_dir()
        saver = Saver(
            output_dir, prefix=prefix, max_to_keep=max_to_keep,
            keep_every_n_hours=keep_every_n_hours)

        now = 0
        for _ in range(20):
            now += 420  # 7 mins
            saver.save(
                {'foo': _gen_random_value()},
                global_step=now/60, now=now
            )

        files = _list_files(prefix, output_dir)
        for i in range(1, 18, 4):
            filepath = os.path.join(
                output_dir, '{}_{}.h5'.format(prefix, 7 * i))
            self.assertTrue(
                filepath in files,
                'File is not found in output dir. {}'.format(filepath))
