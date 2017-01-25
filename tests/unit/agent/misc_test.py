from __future__ import division
from __future__ import absolute_import

import unittest

from luchador.agent.misc import EGreedy


class EGreedyTest(unittest.TestCase):
    def test_anneal(self):
        """Epsilon decrease every time `act_random` is called"""
        e0, et, duration = 1.0, 0.1, 10
        eg = EGreedy(epsilon_init=e0, epsilon_term=et, duration=duration)
        epsilon = e0
        self.assertTrue(eg.epsilon - epsilon < 1e-5)
        for _ in range(duration):
            eg.act_random()
            epsilon += (et - e0) / duration
            self.assertTrue(eg.epsilon - epsilon < 1e-5)
        for _ in range(duration):
            eg.act_random()
            self.assertTrue(eg.epsilon - et < 1e-5)

    def test_anneal_0(self):
        """Epsilon is always e_term if dutaion=0"""
        et = 0.1
        eg = EGreedy(epsilon_init=1.0, epsilon_term=et, duration=0)
        for _ in range(100):
            self.assertEqual(eg.epsilon, et)
