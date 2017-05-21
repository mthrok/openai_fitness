"""Test Layer behaviors"""
from __future__ import division
from __future__ import absolute_import

import numpy as np

# import theano
# theano.config.optimizer = 'None'
# theano.config.exception_verbosity = 'high'

from luchador import nn
from tests.unit.fixture import TestCase

# pylint: disable=invalid-name,too-many-locals


class ReLUTest(TestCase):
    """Test ReLU activation"""
    def test_relu(self):
        """Test ReLU output """
        shape = (3, 4)
        with nn.variable_scope(self.get_scope()):
            in_var = nn.Input(shape=shape)
            layer = nn.layer.ReLU()
            out_var = layer(in_var)

        in_val = np.random.rand(*shape)
        session = nn.Session()

        expected = np.maximum(0, in_val)
        out_val = session.run(
            outputs=out_var,
            inputs={in_var: in_val}
        )
        self.assertEqual(out_var.shape, out_val.shape)
        np.testing.assert_almost_equal(out_val, expected)


class LeakyReLUTest(TestCase):
    """Test LeakyReLU activation"""
    def test_lrelu(self):
        """Test LeakyReLU output """
        alpha, shape = 0.1, (3, 4)
        with nn.variable_scope(self.get_scope()):
            in_var = nn.Input(shape=shape)
            layer = nn.layer.LeakyReLU(alpha=alpha)
            out_var = layer(in_var)

        in_val = np.random.rand(*shape)
        session = nn.Session()

        expected = np.maximum(0, in_val) + np.minimum(0, alpha * in_val)
        out_val = session.run(
            outputs=out_var,
            inputs={in_var: in_val}
        )
        self.assertEqual(out_var.shape, out_val.shape)
        np.testing.assert_almost_equal(out_val, expected)

    def test_lrelu_parameter(self):
        """Parameter retrieval failes when train=False"""
        base_scope, scope, alpha, shape = self.get_scope(), 'foo', 0.1, (3, 4)
        with nn.variable_scope(base_scope):
            in_var = nn.Input(shape=shape)
            layer = nn.layer.LeakyReLU(alpha=alpha, train=False, scope=scope)
            layer(in_var)

        with self.assertRaises(KeyError):
            layer.get_parameter_variable('alpha')

    def test_plrelu_parameter(self):
        """Parameter retrieval succeeds when train=True"""
        base_scope, scope, alpha, shape = self.get_scope(), 'foo', 0.1, (3, 4)
        with nn.variable_scope(base_scope):
            in_var = nn.Input(shape=shape)
            layer = nn.layer.LeakyReLU(alpha=alpha, train=True, scope=scope)
            layer(in_var)

        self.assertIs(
            layer.get_parameter_variable('alpha'),
            nn.get_variable('{}/{}/alpha'.format(base_scope, scope))
        )


class SoftplusTest(TestCase):
    """Test Softplus activation"""
    def test_relu(self):
        """Test Softplus output """
        shape = (3, 4)
        with nn.variable_scope(self.get_scope()):
            in_var = nn.Input(shape=shape)
            layer = nn.layer.Softplus()
            out_var = layer(in_var)

        in_val = np.random.rand(*shape)
        session = nn.Session()

        expected = np.log(1 + np.exp(in_val))
        out_val = session.run(
            outputs=out_var,
            inputs={in_var: in_val}
        )
        self.assertEqual(out_var.shape, out_val.shape)
        np.testing.assert_almost_equal(out_val, expected)


class SigmoidTest(TestCase):
    """Test Sigmoid activation"""
    def test_relu(self):
        """Test Sigmoid output """
        shape = (3, 4)
        with nn.variable_scope(self.get_scope()):
            in_var = nn.Input(shape=shape)
            layer = nn.layer.Sigmoid()
            out_var = layer(in_var)

        in_val = np.random.rand(*shape)
        session = nn.Session()

        expected = 1 / (1 + np.exp(-in_val))
        out_val = session.run(
            outputs=out_var,
            inputs={in_var: in_val}
        )
        self.assertEqual(out_var.shape, out_val.shape)
        np.testing.assert_almost_equal(out_val, expected)


class TanhTest(TestCase):
    """Test Tanh activation"""
    def test_relu(self):
        """Test Tanh output """
        shape = (3, 4)
        with nn.variable_scope(self.get_scope()):
            in_var = nn.Input(shape=shape)
            layer = nn.layer.Tanh()
            out_var = layer(in_var)

        in_val = np.random.rand(*shape)
        session = nn.Session()

        expected = np.tanh(in_val)
        out_val = session.run(
            outputs=out_var,
            inputs={in_var: in_val}
        )
        self.assertEqual(out_var.shape, out_val.shape)
        np.testing.assert_almost_equal(out_val, expected)


class SoftmaxTest(TestCase):
    """Test Softmax activation"""
    def test_relu(self):
        """Test Softmax output """
        shape = (3, 4)
        with nn.variable_scope(self.get_scope()):
            in_var = nn.Input(shape=shape)
            layer = nn.layer.Softmax()
            out_var = layer(in_var)

        in_val = np.random.rand(*shape)
        session = nn.Session()

        expected = np.exp(in_val)
        expected /= np.sum(expected, axis=1, keepdims=True)
        out_val = session.run(
            outputs=out_var,
            inputs={in_var: in_val}
        )
        self.assertEqual(out_var.shape, out_val.shape)
        np.testing.assert_almost_equal(out_val, expected)
