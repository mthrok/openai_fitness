from __future__ import division
from __future__ import absolute_import

import os
import unittest

from luchador.agent import misc


class EGreedyTest(unittest.TestCase):
    def test_anneal(self):
        """Epsilon decrease every time `act_random` is called"""
        e0, et, duration = 1.0, 0.1, 10
        eg = misc.EGreedy(epsilon_init=e0, epsilon_term=et, duration=duration)
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
        eg = misc.EGreedy(epsilon_init=1.0, epsilon_term=et, duration=0)
        for _ in range(100):
            self.assertEqual(eg.epsilon, et)


def _run_plot_test():
    return 'TEST_NOISE' in os.environ


class NoiseTest(unittest.TestCase):
    _fig = None

    @classmethod
    def setUpClass(cls):
        if _run_plot_test():
            import matplotlib.pyplot as plt
            cls.plt = plt
            cls._fig = plt.figure()

    @classmethod
    def tearDownClass(cls):
        if _run_plot_test():
            cls.plt.show()

    @unittest.skipUnless(_run_plot_test(), 'automated test')
    def test_weiner_process(self):
        """Plot Weiner Noise sampling"""
        noise = misc.WienerNoise(shape=(1,), delta=1, dt=1, seed=0)
        samples = [noise.sample() for _ in range(500)]
        ax = self._fig.add_subplot(2, 1, 1)
        ax.plot(samples)
        ax.set_title('Weiner Process')

    @unittest.skipUnless(_run_plot_test(), 'automated test')
    def test_ou_process(self):
        """Plot OU Noise sampling"""
        theta, sigma = 0.15, 0.2
        noise = misc.OUNoise(
            shape=(1,), theta=theta, sigma=sigma, mu=0, seed=0)
        samples = [noise.sample() for _ in range(500)]
        ax = self._fig.add_subplot(2, 1, 2)
        ax.plot(samples)
        ax.set_title(
            'Ornstein-Uhlenbeck Process: theta={}, sigma={}'
            .format(theta, sigma)
        )
