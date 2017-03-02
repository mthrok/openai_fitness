"""Test Layer behaviors"""
from __future__ import division
from __future__ import absolute_import

import numpy as np

# import theano
# theano.config.optimizer = 'None'
# theano.config.exception_verbosity = 'high'

from luchador import nn
from tests.unit.fixture import TestCase


class NormalNoiseTest(TestCase):
    """Test for NormalNoise class"""
    def test_construction(self):
        """NormalNoise layer is built"""
        shape = (4, 5)
        with nn.variable_scope(self.get_scope()):
            noise = nn.layer.NormalNoise()
            in_var = nn.Input(shape=shape, name='original_input')
            out_var = noise(in_var)

        in_val = np.zeros(shape, dtype=in_var.dtype)

        session = nn.Session()
        out_val = session.run(
            outputs=out_var, givens={in_var: in_val})

        self.assertEqual(in_var.shape, out_var.shape)
        self.assertEqual(in_var.shape, out_val.shape)

    def test_ouptut_fetch(self):
        """NormalNoise output is fetched"""
        scope, name = self.get_scope(), 'NormalNoise'
        with nn.variable_scope(scope) as vs:
            input_ = nn.Input(shape=(32, 4, 8, 8), name='input')
            layer = nn.get_layer('NormalNoise')(name=name)
            output = layer(input_)

        with nn.variable_scope(vs, reuse=True):
            self.assertIs(output, nn.get_tensor('{}/output'.format(name)))
            self.assertIs(input_, nn.get_input('input'))

    def test_multiply_mode(self):
        """NormalNoise works in multiply mode"""
        shape, mode = (4, 5), 'multiply'
        with nn.variable_scope(self.get_scope()):
            noise = nn.layer.NormalNoise(mode=mode)
            in_var = nn.Input(shape=shape, name='original_input')
            out_var = noise(in_var)

        in_val = np.zeros(shape, dtype=in_var.dtype)

        session = nn.Session()
        out_val = session.run(
            outputs=out_var, givens={in_var: in_val})

        self.assertEqual(in_var.shape, out_var.shape)
        self.assertEqual(in_var.shape, out_val.shape)
        np.testing.assert_almost_equal(out_val, in_val)


class UniformNoiseTest(TestCase):
    """Test for UniformNoise class"""
    def test_construction(self):
        """UniformNoise layer is built"""
        shape = (4, 5)
        with nn.variable_scope(self.get_scope()):
            noise = nn.layer.UniformNoise()
            in_var = nn.Input(shape=shape, name='original_input')
            out_var = noise(in_var)

        in_val = np.zeros(shape, dtype=in_var.dtype)

        session = nn.Session()
        out_val = session.run(
            outputs=out_var, givens={in_var: in_val})

        self.assertEqual(in_var.shape, out_var.shape)
        self.assertEqual(in_var.shape, out_val.shape)

    def test_ouptut_fetch(self):
        """UniformNoise output is fetched"""
        scope, name = self.get_scope(), 'UniformNoise'
        with nn.variable_scope(scope) as vs:
            input_ = nn.Input(shape=(32, 4, 8, 8), name='input')
            layer = nn.get_layer('UniformNoise')(name=name)
            output = layer(input_)

        with nn.variable_scope(vs, reuse=True):
            self.assertIs(output, nn.get_tensor('{}/output'.format(name)))
            self.assertIs(input_, nn.get_input('input'))

    def test_multiply_mode(self):
        """UniformNoise works in multiply mode"""
        shape, mode = (4, 5), 'multiply'
        with nn.variable_scope(self.get_scope()):
            noise = nn.layer.UniformNoise(mode=mode)
            in_var = nn.Input(shape=shape, name='original_input')
            out_var = noise(in_var)

        in_val = np.zeros(shape, dtype=in_var.dtype)

        session = nn.Session()
        out_val = session.run(
            outputs=out_var, givens={in_var: in_val})

        self.assertEqual(in_var.shape, out_var.shape)
        self.assertEqual(in_var.shape, out_val.shape)
        np.testing.assert_almost_equal(out_val, in_val)
