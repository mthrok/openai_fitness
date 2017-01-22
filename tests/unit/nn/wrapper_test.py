from __future__ import absolute_import

import unittest

import numpy as np

import luchador
from luchador import nn


def _create_ones_tensor(shape):
    if luchador.get_nn_backend() == 'theano':
        import theano.tensor as be
    else:
        import tensorflow as be
    return nn.Tensor(be.ones(shape, dtype='int32'), shape=shape)


class TestTensorOps(unittest.TestCase):
    def test_mul(self):
        constant, shape = 10, (3, 5)
        with nn.variable_scope(self.id().replace('.', '/')):
            output_tensor1 = _create_ones_tensor(shape)
            output_tensor2 = output_tensor1 * constant
            output_tensor3 = constant * output_tensor1

        session = nn.Session()

        output_val1, output_val2, output_val3 = session.run(
            outputs=[output_tensor1, output_tensor2, output_tensor3],
        )
        self.assertTrue(np.all(output_val1 * constant == output_val2))
        self.assertTrue(np.all(output_val1 * constant == output_val3))

    def test_add_numbers(self):
        constant, shape = 10, (3, 5)
        with nn.variable_scope(self.id().replace('.', '/')):
            output_tensor1 = _create_ones_tensor(shape)
            output_tensor2 = output_tensor1 + constant
            output_tensor3 = constant + output_tensor1

        session = nn.Session()

        output_val1, output_val2, output_val3 = session.run(
            outputs=[output_tensor1, output_tensor2, output_tensor3],
        )
        self.assertTrue(np.all(output_val1 + constant == output_val2))
        self.assertTrue(np.all(output_val1 + constant == output_val3))

    def test_add_tensor(self):
        shape = (3, 5)
        with nn.variable_scope(self.id().replace('.', '/')):
            output_tensor1 = _create_ones_tensor(shape)
            output_tensor2 = output_tensor1 + output_tensor1

        session = nn.Session()
        output_val1, output_val2 = session.run(
            outputs=[output_tensor1, output_tensor2],
        )
        self.assertTrue(np.all(output_val1 + output_val1 == output_val2))

    def test_add_input(self):
        shape = (3, 5)
        with nn.variable_scope(self.id().replace('.', '/')):
            input_variable = nn.Input(shape=shape, dtype='int32')()
            output_tensor1 = _create_ones_tensor(shape)
            output_tensor2 = output_tensor1 + input_variable

        session = nn.Session()
        input_val = 10 * np.ones(shape, dtype='int32')
        output_val1, output_val2 = session.run(
            outputs=[output_tensor1, output_tensor2],
            givens={input_variable: input_val}
        )
        self.assertTrue(np.all(output_val1 + input_val == output_val2))

    def test_neg(self):
        shape = (3, 5)
        with nn.variable_scope(self.id().replace('.', '/')):
            output_tensor1 = _create_ones_tensor(shape)
            output_tensor2 = -output_tensor1

        session = nn.Session()
        output_val1, output_val2 = session.run(
            outputs=[output_tensor1, output_tensor2],
        )
        self.assertTrue(np.all(-output_val1 == output_val2))
