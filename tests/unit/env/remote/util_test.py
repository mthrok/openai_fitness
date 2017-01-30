"""Test env.remote.util module"""
from __future__ import absolute_import

import unittest

import numpy as np

from luchador.env import Outcome
from luchador.env.remote import util


def _test_round_trip(expected, compress):
    # pylint: disable=protected-access
    actual = util._deserialize_state(
        util._serialize_state(expected, compress=compress)
    )
    np.testing.assert_equal(expected, actual)


class TestArraySerialization(unittest.TestCase):
    """Test if deserialization recovers the original state"""
    # pylint: disable=no-self-use
    def test_serialize_int(self):
        """Int array is deserialized to original state"""
        expected = np.random.randint(0, 2**8, size=(200, 300), dtype='uint8')
        _test_round_trip(expected, compress=False)

    def test_serialize_int_compress(self):
        """Int array is deserialized to original state"""
        expected = np.random.randint(0, 2**8, size=(200, 300), dtype='uint8')
        _test_round_trip(expected, compress=True)

    def test_serialize_float(self):
        """Int array is deserialized to original state"""
        expected = np.random.rand(200, 300).astype('float32')
        _test_round_trip(expected, compress=False)

    def test_serialize_float_compress(self):
        """Int array is deserialized to original state"""
        expected = np.random.rand(200, 300).astype('float32')
        _test_round_trip(expected, compress=True)


class TestOutcomeSerialization(unittest.TestCase):
    """Test if Outcome recovers the original state"""
    def test_array_serialization(self):
        """NumPy NDArray is correctly serialized"""
        reward = 1.0
        state = np.random.randint(0, 256, size=(3, 84, 84), dtype='uint8')
        terminal = True
        info = {}
        outcome = Outcome(
            reward=reward, state=state, terminal=terminal, info=info)
        obj = util.serialize_outcome(outcome)
        outcome2 = util.deserialize_outcome(obj)

        self.assertEqual(outcome.reward, outcome2.reward)
        np.testing.assert_equal(outcome.state, outcome2.state)
        self.assertEqual(outcome.terminal, outcome2.terminal)
        self.assertEqual(outcome.info, outcome2.info)

    def test_pod_serialization(self):
        """Primitive type is correctly serialized"""
        reward = 3
        state = [1, 2, 3]
        terminal = False
        info = {}
        outcome = Outcome(
            reward=reward, state=state, terminal=terminal, info=info)
        obj = util.serialize_outcome(outcome)
        outcome2 = util.deserialize_outcome(obj)

        self.assertEqual(outcome.reward, outcome2.reward)
        self.assertEqual(outcome.state, outcome2.state)
        self.assertEqual(outcome.terminal, outcome2.terminal)
        self.assertEqual(outcome.info, outcome2.info)
