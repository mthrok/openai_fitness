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

    np.testing.assert_almost_equal(output_val_, output_val)


class AnonymousSingleInputTest(TestCase):
    """Test for Anonyomus class"""
    def test_neg(self):
        """Anonymous layer can handle negation"""
        input_val = np.random.rand(3, 4)
        output_val = -input_val
        _test('-x', input_val, output_val, self.get_scope())

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
