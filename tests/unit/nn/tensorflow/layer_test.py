from __future__ import absolute_import

import unittest

import luchador
from luchador import nn


@unittest.skipUnless(
    luchador.get_nn_backend() == 'tensorflow', 'Tensorflow backend')
class TestConv2D(unittest.TestCase):
    longMessage = True

    def test_map_padding(self):
        """padding string is correctly mapped to valid one"""
        # pylint: disable=protected-access

        inputs = ('full', 'half', 'same', 'valid')
        expecteds = ('VALID', 'SAME', 'SAME', 'VALID')
        for i, expected in zip(inputs, expecteds):
            found = nn.layer.convolution._map_padding(i)
            self.assertEqual(expected, found)
