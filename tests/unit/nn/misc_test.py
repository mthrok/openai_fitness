from __future__ import absolute_import

import unittest

import numpy as np

from luchador import nn
from tests.unit import fixture


class TestMisc(unittest.TestCase):
    def _test_mean(self, axis, shape, keep_dims):
        with nn.variable_scope(self.id().replace('.', '/')):
            tensor0 = fixture.create_tensor(shape)
            tensor1 = nn.mean(tensor0, axis=axis, keep_dims=keep_dims)

        session = nn.Session()

        val0, val1 = session.run(
            outputs=[tensor0, tensor1],
        )
        expected = val0.mean(axis=axis, keepdims=keep_dims)
        self.assertTrue(np.all(val1 == expected))

    def test_mean(self):
        """Test mean with single axis, dropping axis"""
        self._test_mean(0, (3, 5), False)

    def test_mean_keep_dim(self):
        """Test mean with single axis, dropping axis"""
        self._test_mean(0, (3, 5), True)

    def test_mean_multi(self):
        """Test mean with multiple axes, dropping axis"""
        self._test_mean((1, 2), (3, 4, 5, 6), False)

    def test_mean_multi_keep_dim(self):
        """Test mean with multiple axes, dropping axis"""
        self._test_mean((1, 2), (3, 4, 5, 6), True)
