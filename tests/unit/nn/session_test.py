from __future__ import absolute_import

import unittest

import numpy as np

from luchador import nn


class SessionTest(unittest.TestCase):
    def _test_load_dataset(self, dtype1, dtype2):
        name = 'test_load_dataset_{}_{}'.format(dtype1, dtype2)
        shape = (3, 3)
        target_value = 10

        variable = nn.get_variable(name=name, shape=shape, dtype=dtype1)
        value = target_value * np.ones(shape, dtype=dtype2)

        session = nn.Session()
        session.load_dataset({name: value}, cast=not dtype1 == dtype2)

        updated_value = session.run(outputs=variable)
        self.assertTrue(np.all(target_value == updated_value))

    def test_load_dataset_float32(self):
        """Variable value is set via Session.load_dataset when dtype=float32"""
        self._test_load_dataset('float32', 'float32')

    def test_load_dataset_float64(self):
        """Variable value is set via Session.load_dataset when dtype=float64"""
        self._test_load_dataset('float64', 'float64')

    def test_load_dataset_downcast(self):
        """Variable value is set when data is downcasted"""
        self._test_load_dataset('float32', 'float64')

    def test_load_dataset_upcast(self):
        """Variable value is set when data is downcasted"""
        self._test_load_dataset('float64', 'float32')
