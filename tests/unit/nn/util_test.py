from __future__ import absolute_import

import unittest

from luchador.nn.util import (
    get_model_config,
    make_model,
)


class UtilTest(unittest.TestCase):
    longMessage = True
    maxDiff = None

    def test_create_model(self):
        """Deserialized model is equal to the original"""
        cfg1 = get_model_config('vanilla_dqn', n_actions=5)
        m1 = make_model(cfg1)
        m2 = make_model(m1.serialize())
        self.assertEqual(m1, m2)
