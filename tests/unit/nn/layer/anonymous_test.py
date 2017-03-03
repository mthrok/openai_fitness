"""Test Layer behaviors"""
from __future__ import division
from __future__ import absolute_import

import numpy as np

# import theano
# theano.config.optimizer = 'None'
# theano.config.exception_verbosity = 'high'

from luchador import nn
from tests.unit.fixture import TestCase


def _test(exp, input_val, output_val, scope):
    """Run Anonymous layer and check result"""
    input_var = nn.Input(shape=input_val.shape, dtype=input_val.dtype)
    with nn.variable_scope(scope):
        layer = nn.layer.Anonymous(exp)
        output_var = layer(input_var)

    session = nn.Session()
    output_val_ = session.run(
        outputs=output_var, inputs={input_var: input_val})

    np.testing.assert_almost_equal(output_val_, output_val, decimal=4)


class AnonymousSingleInputTest(TestCase):
    """Test for Anonyomus class"""
    def test_neg(self):
        """Anonymous layer can handle negation"""
        input_val = np.random.rand(3, 4)
        output_val = -input_val
        _test('-x', input_val, output_val, self.get_scope())

    def test_abs(self):
        """Anonymous layer can handle abs"""
        input_val = np.random.rand(3, 4)
        output_val = abs(input_val)
        _test('abs(x, name="output")', input_val, output_val, self.get_scope())

    def test_add(self):
        """Anonymous layer can handle addition"""
        input_val = np.random.rand(3, 4)
        output_val = 10 + input_val
        _test('10 + x', input_val, output_val, self.get_scope())
        _test('x + 10', input_val, output_val, self.get_scope())

    def test_sub(self):
        """Anonymous layer can handle subtraction"""
        input_val = np.random.rand(3, 4)
        output_val = 10 - input_val
        _test('10 - x', input_val, output_val, self.get_scope())
        _test('x - 10', input_val, -output_val, self.get_scope())

    def test_multi(self):
        """Anonymous layer can handle multiplication"""
        input_val = np.random.rand(3, 4)
        output_val = 2 * input_val
        _test('2 * x', input_val, output_val, self.get_scope())
        _test('x * 2', input_val, output_val, self.get_scope())

    def test_div(self):
        """Anonymous layer can handle division"""
        input_val = np.random.rand(3, 4)
        output_val = input_val / 7
        _test('x / 7', input_val, output_val, self.get_scope())

    def test_exp(self):
        """Anonymous layer can handle exponential"""
        input_val = np.random.rand(3, 4)
        output_val = np.exp(input_val)
        _test('exp(x)', input_val, output_val, self.get_scope())

    def test_log(self):
        """Anonymous layer can handle log"""
        input_val = np.random.rand(3, 4)
        output_val = np.log(input_val)
        _test('log(x)', input_val, output_val, self.get_scope())

    def test_sin(self):
        """Anonymous layer can handle sin"""
        input_val = np.random.rand(3, 4)
        output_val = np.sin(input_val)
        _test('sin(x)', input_val, output_val, self.get_scope())

    def test_cos(self):
        """Anonymous layer can handle cos"""
        input_val = np.random.rand(3, 4)
        output_val = np.cos(input_val)
        _test('cos(x)', input_val, output_val, self.get_scope())

    def test_mean_single(self):
        """Anonymous layer can handle mean"""
        input_val = np.random.rand(3, 4, 5, 6)
        output_val = input_val.mean(axis=1)
        _test('x.mean(axis=1)', input_val, output_val, self.get_scope())

    def test_mean_multi(self):
        """Anonymous layer can handle cos"""
        input_val = np.random.rand(3, 4, 5, 6)
        output_val = np.mean(input_val, axis=(1, 2))
        _test('x.mean(axis=(1, 2))', input_val, output_val, self.get_scope())

    def test_mean_all(self):
        """Anonymous layer can handle cos"""
        input_val = np.random.rand(3, 4, 5, 6)
        output_val = np.mean(input_val)
        _test('x.mean()', input_val, output_val, self.get_scope())

    def test_reshape(self):
        """Anonymous layer can handle reshape"""
        input_val = np.random.rand(3, 4)
        output_val = input_val.reshape((-1, 1))
        _test('x.reshape((-1, 1))', input_val, output_val, self.get_scope())

    def test_tile(self):
        """Anonymous layer can handle tile"""
        input_val = np.random.rand(3, 4)
        output_val = np.tile(input_val, (1, 3, 5))
        _test('x.tile((1, 3, 5))', input_val, output_val, self.get_scope())

    def test_max(self):
        """Anonymous layer can handle max"""
        input_val = np.random.rand(3, 4)
        output_val = input_val.max(axis=1)
        _test('x.max(axis=1)', input_val, output_val, self.get_scope())

    def test_sum(self):
        """Anonymous layer can handle sum"""
        input_val = np.random.rand(3, 4)
        output_val = input_val.sum(axis=1)
        _test('x.sum(axis=1)', input_val, output_val, self.get_scope())

    def test_mean_shift(self):
        """Anonymous layer can handle complex arithmetic"""
        input_val = np.random.rand(3, 4)
        mean_ = np.tile(input_val.mean(axis=1, keepdims=True), (1, 4))
        output_val = input_val - mean_
        _test(
            'x - x.mean(axis=1, keep_dims=True).tile((1, 4))',
            input_val, output_val, self.get_scope()
        )

    def test_norm(self):
        """Anonymous layer can handle complex arithmetic"""
        input_val = np.random.rand(3, 4)
        output_val = (input_val * input_val).sum()
        _test('(x * x).sum()', input_val, output_val, self.get_scope())
