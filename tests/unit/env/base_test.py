from __future__ import absolute_import

import unittest

import numpy as np

import luchador.env


class TestOutcome(unittest.TestCase):
    def test_numpy_serialization(self):
        """NumPy NDArray is correctly serialized"""
        reward = 1.0
        obs = np.random.randint(0, 256, size=(3, 84, 84), dtype='uint8')
        terminal = True
        state = {}
        outcome = luchador.env.Outcome(reward, obs, terminal, state)
        obj = luchador.env.serialize_outcome(outcome)
        outcome2 = luchador.env.deserialize_outcome(obj)

        self.assertEqual(outcome.reward, outcome2.reward)
        self.assertTrue(np.all(outcome.observation == outcome2.observation))
        self.assertEqual(outcome.terminal, outcome2.terminal)
        self.assertEqual(outcome.state, outcome2.state)

    def test_pod_serialization(self):
        """Primitive type is correctly serialized"""
        reward = 3
        obs = [1, 2, 3]
        terminal = False
        state = {}
        outcome = luchador.env.Outcome(reward, obs, terminal, state)
        obj = luchador.env.serialize_outcome(outcome)
        outcome2 = luchador.env.deserialize_outcome(obj)

        self.assertEqual(outcome.reward, outcome2.reward)
        self.assertEqual(outcome.observation, outcome2.observation)
        self.assertEqual(outcome.terminal, outcome2.terminal)
        self.assertEqual(outcome.state, outcome2.state)
