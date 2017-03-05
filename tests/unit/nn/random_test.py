"""Test Layer behaviors"""
from __future__ import division
from __future__ import absolute_import

import numpy as np

# import theano
# theano.config.optimizer = 'None'
# theano.config.exception_verbosity = 'high'

from luchador import nn
from tests.unit.fixture import TestCase


class _NoiseTest(TestCase):
    longMessage = True

    def _validate(self, in_var, in_val, mean, std, out_var):
        session = nn.Session()
        out_val = session.run(outputs=out_var, givens={in_var: in_val})

        self.assertEqual(in_var.shape, out_var.shape)
        self.assertEqual(in_var.dtype, out_var.dtype)

        self.assertEqual(in_var.shape, out_val.shape)
        self.assertEqual(in_var.dtype, out_val.dtype)

        mean_diff = abs(np.mean(out_val) - mean)
        std_diff = abs(np.std(out_val) - std)
        self.assertLess(mean_diff, 0.3)
        self.assertLess(std_diff, 0.3)

    def _test_add(self, noise, shape, mean, std, scope):
        with nn.variable_scope(scope):
            in_var = nn.Input(shape=shape, name='original_input')
            out_var_1 = noise + in_var
            out_var_2 = in_var + noise
        in_val = np.zeros(shape=in_var.shape, dtype=in_var.dtype)
        self._validate(in_var, in_val, mean, std, out_var_1)
        self._validate(in_var, in_val, mean, std, out_var_2)

    def _test_mult(self, noise, shape, mean, std, scope):
        with nn.variable_scope(scope):
            in_var = nn.Input(shape=shape, name='original_input')
            out_var_1 = noise * in_var
            out_var_2 = in_var * noise
        in_val = np.ones(shape=in_var.shape, dtype=in_var.dtype)
        self._validate(in_var, in_val, mean, std, out_var_1)
        self._validate(in_var, in_val, mean, std, out_var_2)

    def _test_sub(self, noise, shape, mean, std, scope):
        with nn.variable_scope(scope):
            in_var = nn.Input(shape=shape, name='original_input')
            out_var_1 = noise - in_var
            out_var_2 = in_var - noise
        in_val = 10 * np.ones(shape=in_var.shape, dtype=in_var.dtype)
        self._validate(in_var, in_val, mean - 10, std, out_var_1)
        self._validate(in_var, in_val, 10 - mean, std, out_var_2)


class NormalRandomTest(_NoiseTest):
    """Test for NormalRandom class"""
    def test_noise_add(self):
        """NormalNoise can be added to Tensor"""
        shape, mean, std = (10000,), 3, 5
        noise = nn.NormalRandom(mean=mean, std=std)
        self._test_add(noise, shape, mean, std, self.get_scope())

    def test_noise_sub(self):
        """NormalNoise can be subed to/from Tensor"""
        shape, mean, std = (10000,), 3, 5
        noise = nn.NormalRandom(mean=mean, std=std)
        self._test_sub(noise, shape, mean, std, self.get_scope())

    def test_noise_mult(self):
        """NormalNoise can be multiplied to Tensor"""
        shape, mean, std = (10000,), 3, 5
        noise = nn.NormalRandom(mean=mean, std=std)
        self._test_mult(noise, shape, mean, std, self.get_scope())


class UniformRandomTest(_NoiseTest):
    """Test for UniformRandom class"""
    def test_noise_add(self):
        """UniformNoise can be added to Tensor"""
        shape, low, high = (10000,), 3, 5
        mean, std = (low + high) / 2, (high - low) / np.sqrt(12)
        noise = nn.UniformRandom(low=low, high=high)
        self._test_add(noise, shape, mean, std, self.get_scope())

    def test_noise_sub(self):
        """UniformNoise can be subed to/from Tensor"""
        shape, low, high = (10000,), 3, 5
        mean, std = (low + high) / 2, (high - low) / np.sqrt(12)
        noise = nn.UniformRandom(low=low, high=high)
        self._test_sub(noise, shape, mean, std, self.get_scope())

    def test_noise_mult(self):
        """UniformNoise can be multiplied to Tensor"""
        shape, low, high = (10000,), 3, 5
        mean, std = (low + high) / 2, (high - low) / np.sqrt(12)
        noise = nn.UniformRandom(low=low, high=high)
        self._test_mult(noise, shape, mean, std, self.get_scope())
