"""Test nn.model.util module"""
from __future__ import absolute_import

import unittest

import luchador
from luchador import nn


class UtilTest(unittest.TestCase):
    """Test model [de]serialization"""
    longMessage = True
    maxDiff = None

    def test_create_model(self):
        """Deserialized model is equal to the original"""
        fmt = luchador.get_nn_conv_format()
        shape = '[null, 4, 84, 84]' if fmt == 'NCHW' else '[null, 84, 84, 4]'
        cfg1 = nn.get_model_config(
            'vanilla_dqn', n_actions=5, input_shape=shape)
        m1 = nn.make_model(cfg1)
        m2 = nn.make_model(m1.serialize())
        self.assertEqual(m1, m2)
