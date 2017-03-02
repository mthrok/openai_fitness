from __future__ import absolute_import

import unittest

import luchador

_BE = luchador.get_nn_backend()


@unittest.skipUnless(_BE == 'tensorflow', 'Tensorflow backend')
class TestConv2D(unittest.TestCase):
    longMessage = True

    def test_map_padding(self):
        """padding string is correctly mapped to valid one"""
        from luchador.nn.core.backend.tensorflow.layer import convolution
        # pylint: disable=protected-access

        inputs = ('full', 'half', 'same', 'valid')
        expecteds = ('VALID', 'SAME', 'SAME', 'VALID')
        for i, expected in zip(inputs, expecteds):
            found = convolution._map_padding(i)
            self.assertEqual(expected, found)
