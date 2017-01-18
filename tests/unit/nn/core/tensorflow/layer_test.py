from __future__ import absolute_import

import unittest

import luchador


@unittest.skipUnless(
    luchador.get_nn_backend() == 'tensorflow', 'Tensorflow backend')
class TestConv2D(unittest.TestCase):
    longMessage = True

    def test_map_padding(self):
        """padding string is correctly mapped to valid one"""
        import luchador.nn.core.tensorflow.layer as layer_module
        # pylint: disable=protected-access

        inputs = ('full', 'half', 'same', 'valid')
        expecteds = ('VALID', 'SAME', 'SAME', 'VALID')
        for i, expected in zip(inputs, expecteds):
            found = layer_module._map_padding(i)
            self.assertEqual(expected, found)
